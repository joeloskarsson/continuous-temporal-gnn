import torch
import torch.nn as nn

from models.gru_decay import GRUDecayCell
import utils

# Handles all nodes together in a single GRU-unit
class GRUModel(nn.Module):
    def __init__(self, config):
        super(GRUModel, self).__init__()

        self.time_input = bool(config["time_input"])
        self.mask_input = bool(config["mask_input"])
        self.has_features = config["has_features"]

        output_dim = self.compute_output_dim(config)

        self.gru_cells = self.create_cells(config)

        if config["learn_init_state"]:
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["hidden_dim"])
        else:
            self.init_state_param = torch.zeros(config["gru_layers"],
                    config["hidden_dim"], device=config["device"])

        first_post_dim = self.compute_pred_input_dim(config)
        if config["n_fc"] == 1:
            fc_layers = [nn.Linear(first_post_dim, output_dim)]
        else:
            fc_layers = []
            for layer_i in range(config["n_fc"]-1):
                fc_layers.append(nn.Linear(first_post_dim if (layer_i == 0)
                    else config["hidden_dim"], config["hidden_dim"]))
                fc_layers.append(nn.ReLU())

            fc_layers.append(nn.Linear(config["hidden_dim"], output_dim))

        self.post_gru_layers = nn.Sequential(*fc_layers)

        self.y_shape = (config["time_steps"], -1, config["num_nodes"]*config["y_dim"])
        self.delta_t_shape = (config["time_steps"], -1, config["num_nodes"])
        self.pred_shape = (config["time_steps"], -1, config["y_dim"]) # Returned shape
        self.f_shape = (config["time_steps"], -1,
                config["num_nodes"]*config["feature_dim"])

        self.init_decay_weight = torch.zeros(config["hidden_dim"],
                device=config["device"])

    def create_cells(self, config):
        input_dim = self.compute_gru_input_dim(config)

        return nn.ModuleList([
                GRUDecayCell(input_dim if layer_i==0 else config["hidden_dim"], config)
            for layer_i in range(config["gru_layers"])])

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["num_nodes"]*(config["y_dim"] + config["feature_dim"] +
                int(self.mask_input)+ int(self.time_input))  # Add N if delta_t input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["num_nodes"]*(config["feature_dim"] +
                int(self.time_input)) # Add N if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["num_nodes"]*config["y_dim"]

    def get_init_states(self, num_graphs):
        return self.init_state_param.unsqueeze(1).repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B, d_h)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B, N*d_y)

        obs_mask = batch.mask.transpose(0,1).view(self.delta_t_shape) # (N_T, B*N, 1)

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update
        fc_input_tensors = [] # input to fc layers used for prediction

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, B, N*d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_input_tensors.append(input_f_reshaped)

        if self.time_input:
            delta_time_inputs = batch.delta_t.transpose(0,1).view(
                self.delta_t_shape) # (N_T, B, N)

            # Concatenated delta_t to input
            gru_input_tensors.append(delta_time_inputs)
            fc_input_tensors.append(delta_time_inputs)

        if self.mask_input:
            gru_input_tensors.append(obs_mask)

        init_states = self.get_init_states(batch.num_graphs)

        decay_delta_ts = utils.t_to_delta_t(batch.t).transpose(
                0,1).unsqueeze(-1) # (B, N_T)

        gru_input = torch.cat(gru_input_tensors, dim=-1)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            hidden_state = init_state
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            decayed_states = [] # Previous states decayed to obs. time
            hidden_states = [] # New states after observation
            for input_slice, delta_time_slice in zip(gru_input, decay_delta_ts):
                # Always update hidden state, as at least one node is always observed
                decayed_state, hidden_state, decay_target, decay_weight = gru_cell(
                        input_slice, hidden_state, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                decayed_states.append(decayed_state)
                hidden_states.append(hidden_state)

            gru_input = hidden_states

        decayed_states_tensor = torch.stack(decayed_states, dim=0)
        fc_input_tensors.append(decayed_states_tensor)

        predictions = self.post_gru_layers(torch.cat(
            fc_input_tensors, dim=-1)) # Shape (N_T, B, N)
        return predictions.view(*self.pred_shape)

