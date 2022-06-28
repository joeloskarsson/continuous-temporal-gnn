import torch

from models.gru_model import GRUModel
import utils

# Handles each node independently with GRU-units
class GRUNodeModel(GRUModel):
    def __init__(self, config):
        super(GRUNodeModel, self).__init__(config)

        self.num_nodes = config["num_nodes"]
        self.y_shape = (config["time_steps"], -1, config["y_dim"])
        self.f_shape = (config["time_steps"], -1, config["feature_dim"])
        self.state_updates = config["state_updates"]
        assert config["state_updates"] in ("all", "obs", "hop"), (
                f"Unknown state update: {config['state_updates']}")

        # If node-specific initial states should be used
        self.node_init_states = (config["node_params"] and config["learn_init_state"])
        if self.node_init_states:
            # Override initial GRU-states
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["num_nodes"], config["hidden_dim"])

    def get_init_states(self, num_graphs):
        if self.node_init_states:
            return self.init_state_param.repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)
        else:
            return self.init_state_param.unsqueeze(1).repeat(
                1, self.num_nodes*num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["y_dim"] + config["feature_dim"] +\
            int(self.time_input) + int(self.mask_input) # Add one if delta_t/mask input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["feature_dim"] +\
                int(self.time_input) # Add one if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["y_dim"]

    def compute_predictions(self, pred_input, edge_index, edge_weight):
        # pred_input: (N_T, B*N, pred_input_dim)
        return self.post_gru_layers(pred_input) # Shape (N_T, B*N, d_y)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B*N, d_y)

        delta_time = batch.delta_t.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        if self.state_updates == "all":
            # Deltas are between each t, as all nodes are updated
            update_delta_time = utils.t_to_delta_t(batch.t).transpose(0,1)\
                .unsqueeze(2).repeat_interleave(self.num_nodes, dim=1) # (N_T, B*N, 1)
        else:
            # ("obs"/"hop")Update deltas are the ones in batch
            update_delta_time = batch.update_delta_t.transpose(
                0,1).unsqueeze(-1) # (N_T, B*N, 1)

        obs_mask = batch.mask.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        if self.state_updates == "hop":
            update_mask = batch.hop_mask.transpose(0,1).unsqueeze(-1) # (N_T, B*N, 1)
        else:
            # "obs" (or "all", but then unused)
            update_mask = obs_mask

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update
        fc_input_tensors = [] # input to fc layers used for prediction

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, B, N*d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_input_tensors.append(input_f_reshaped)

        if self.time_input:
            # Concatenated delta_t to input
            gru_input_tensors.append(delta_time)
            fc_input_tensors.append(delta_time)

        if self.mask_input:
            # Concatenated mask to input (does not always make sense)
            gru_input_tensors.append(obs_mask)
            # Mask should not be in fc_input, we don't
            # know what will be observed when predicting

        init_states = self.get_init_states(batch.num_graphs)

        gru_input = torch.cat(gru_input_tensors, dim=-1)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            hidden_state = init_state
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            decayed_states = [] # Previous states decayed to obs. time
            hidden_states = [] # New states after observation
            for input_slice, delta_time_slice, update_mask_slice in zip(
                    gru_input, update_delta_time, update_mask):
                decayed_state, new_hidden_state, new_decay_target, new_decay_weight =\
                    gru_cell(input_slice, hidden_state, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                if self.state_updates == "all":
                    # Update for all nodes
                    hidden_state = new_hidden_state
                    decay_target = new_decay_target
                    decay_weight = new_decay_weight
                else:
                    # Only update state for observed nodes
                    hidden_state = update_mask_slice*new_hidden_state +\
                        (1. - update_mask_slice)*hidden_state
                    decay_target = update_mask_slice*new_decay_target +\
                        (1. - update_mask_slice)*decay_target
                    decay_weight = update_mask_slice*new_decay_weight +\
                        (1. - update_mask_slice)*decay_weight

                decayed_states.append(decayed_state)
                hidden_states.append(hidden_state)

            gru_input = hidden_states

        decayed_states_tensor = torch.stack(decayed_states, dim=0)
        fc_input_tensors.append(decayed_states_tensor)

        pred_input = torch.cat(fc_input_tensors, dim=-1) # (N_T, B*N, pred_input_dim)
        return self.compute_predictions(pred_input, batch.edge_index, edge_weight)

