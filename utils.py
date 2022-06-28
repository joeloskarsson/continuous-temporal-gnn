import os
import torch
import json
import pickle
import torch_geometric as ptg
import torch.nn as nn
import numpy as np

import constants

def save_data(name, config, objects):
    os.makedirs(constants.DS_DIR, exist_ok=True)

    ds_dir_path = os.path.join(constants.DS_DIR, name)
    os.makedirs(ds_dir_path, exist_ok=True)

    data_path = os.path.join(ds_dir_path, constants.DATA_FILE_NAME)
    config_path = os.path.join(ds_dir_path, constants.CONFIG_FILE_NAME)

    with open(data_path, "wb") as data_file:
        pickle.dump(objects, data_file)

    # Dump cmd-line arguments as json in dataset directory
    json_string = json.dumps(config, sort_keys=True, indent=4)
    with open(config_path, "w") as json_file:
        json_file.write(json_string)

def load_data(ds_name):
    data_path = os.path.join(constants.DS_DIR, ds_name, constants.DATA_FILE_NAME)
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

    return data

def load_config(ds_name):
    config_path = os.path.join(constants.DS_DIR, ds_name, constants.CONFIG_FILE_NAME)
    with open(config_path, "r") as config_file:
        config_json = json.load(config_file)

    return config_json

def to_tensor(a, device, dtype=torch.float32):
    if type(a) == torch.Tensor:
        return a.to(device=device, dtype=dtype)
    else:
        return torch.tensor(a, device=device, dtype=dtype)

def load_temporal_graph_data(ds_name, batch_size, compute_hop_mask=False, L_hop=1):
    # Always load into CPU memory here
    device = torch.device("cpu")

    data_dict = load_data(ds_name)

    # If full ptg graphs should be loaded
    load_graphs = ("graphs" in data_dict["train"])

    if not load_graphs:
        edge_index = to_tensor(data_dict["edge_index"], device=device, dtype=torch.long)

        if "edge_weight" in data_dict:
            edge_weight = to_tensor(data_dict["edge_weight"], device=device)
        else:
            edge_weight = torch.ones(edge_index.shape[1], device=device)
        # Encode as edge attributes for correct batching
        edge_attr = edge_weight.unsqueeze(1) # Shape (N_edges, 1)

    loaders = []
    for subset in ("train", "val", "test"):
        if load_graphs:
            graphs = data_dict[subset]["graphs"]
        else:
            y = to_tensor(data_dict[subset]["y"], device) # (N_subset, N_T, N, 1)
            t = to_tensor(data_dict[subset]["t"], device) # (N_subset, N_T)
            if "mask" in data_dict[subset]:
                delta_t = to_tensor(data_dict[subset]["delta_t"],
                        device) # (N_subset, N, N_T)
                mask = to_tensor(data_dict[subset]["mask"], device) # (N_subset, N, N_T)
            else:
                delta_t = t_to_delta_t(t).unsqueeze(1).repeat(
                        1,y.shape[2],1) # (N_subset, N, N_T)
                mask = torch.ones_like(delta_t)

            load_features = ("features" in data_dict[subset])
            if load_features:
                features = to_tensor(data_dict[subset]["features"],
                        device) # (N_subset, N_T, N, d_f)
                features = features.transpose(1,2) # Shape (N_subset, N, N_T, d_f)

            num_nodes = y.shape[2]
            y = y.transpose(1,2) # Shape (N_subset, N, N_T, d_y)
            if compute_hop_mask:
                print("Pre-computing L-hop masks...")
                # Compute L hop mask
                neighborhood_list = []
                for node_i in range(num_nodes):
                    hop_neighbors = ptg.utils.k_hop_subgraph(node_i, L_hop,
                            edge_index)[0]
                    node_hop_mask = torch.zeros(num_nodes)
                    node_hop_mask[hop_neighbors] = 1
                    neighborhood_list.append(node_hop_mask)

                hop_neighbor_mask = torch.stack(neighborhood_list).to(
                        bool) # (N,N) where i,j is if j is in L-hop graph of i

                hop_mask_samples = []
                for sample_mask in mask.to(bool): # (N, N_T)
                    hop_mask_samples.append(torch.stack([
                        torch.any(hop_neighbor_mask[time_mask], dim=0)
                        for time_mask in sample_mask.T], dim=1)) # (N, N_T)

                hop_mask = torch.stack(hop_mask_samples, dim=0).to(
                        torch.float32) # (N_samples, N, N_T)
                print("done")

                # Update_delta_t is delta_t, but for state updates instead of obs
                update_delta_t = node_t_deltas(t, hop_mask.transpose(1,2)).transpose(1,2)
            else:
                hop_mask = mask
                update_delta_t = delta_t

            # Create ptg graphs
            if load_features:
                graphs = [ptg.data.Data(edge_index=edge_index, # Shape (2, N_edges)
                            edge_attr = edge_attr, # Shape (N_edges, 1)
                            y=y_sample, # Shape (N, N_T, d_y)
                            features=features_sample, # (N, N_T, d_f)
                            t=t_sample.unsqueeze(0), # Shape (1, N_T)
                            delta_t=delta_t_sample, # Shape (N, N_T)
                            update_delta_t=update_delta_t_sample, # (N, N_T)
                            mask=mask_sample, # (N, N_T)
                            hop_mask=hop_mask_sample, # (N, N_T)
                            num_nodes=num_nodes)
                        for y_sample, features_sample, t_sample,
                            delta_t_sample, mask_sample, hop_mask_sample,
                            update_delta_t_sample
                            in zip(y, features, t, delta_t, mask,
                                hop_mask, update_delta_t)]
            else:
                graphs = [ptg.data.Data(edge_index=edge_index, # Shape (2, N_edges)
                            edge_attr = edge_attr, # Shape (N_edges, 1)
                            y=y_sample, # Shape (N, N_T, d_y)
                            t=t_sample.unsqueeze(0), # Shape (1, N_T)
                            delta_t=delta_t_sample, # Shape (N, N_T)
                            update_delta_t=update_delta_t_sample, # (N, N_T)
                            mask=mask_sample, # (N, N_T)
                            hop_mask=hop_mask_sample, # (N, N_T)
                            num_nodes=num_nodes)
                        for y_sample, t_sample, delta_t_sample,
                            mask_sample, hop_mask_sample, update_delta_t_sample
                        in zip(y, t, delta_t, mask, hop_mask, update_delta_t)]

        loader = ptg.loader.DataLoader(graphs, batch_size=batch_size,
                shuffle=(subset == "train"), num_workers=4, pin_memory=True,
                persistent_workers=True)
        loaders.append(loader)

    return loaders

def new_param(*shape):
    # Same weight init as other torch params
    rand_tensor = 2*torch.rand(shape) - 1 # U([-1,1])
    # Assume last dim is input dim
    return torch.nn.Parameter(rand_tensor/(rand_tensor.shape[-1]**0.5))

def t_to_delta_t(ts):
    # ts has shape (B, N_t)
    return torch.cat((ts[:,:1], (ts[:,1:] - ts[:,:-1])), dim=1) # Shape (B, N_t)

def eval_prediction(pred, target, obs_mask, num_nodes):
    # pred: (N_T, BN, d_y)
    # target: (N_T, BN, d_y)
    # obs_mask: (N_T, BN, 1)
    N_T, _, d_y = pred.shape

    pred = pred.view(N_T, -1, num_nodes, d_y) # (N_T, B, N, d_y)
    target = target.view(N_T, -1, num_nodes, d_y) # (N_T, B, N, d_y)
    obs_mask = obs_mask.view(N_T, -1, num_nodes, 1) # (N_T, B, N, 1)

    dim_errors = {"mse": [], "mae": [], "rmse": [], "max_mse": [], "max_mae": []}
    for y_dim in range(d_y):
        # Prediction and target for specific dimensions
        dim_pred = pred[..., y_dim].unsqueeze(-1) # (N_T, B, N, 1)
        dim_target = target[..., y_dim].unsqueeze(-1) # (N_T, B, N, 1)

        node_obs = torch.sum(obs_mask, dim=(0,1,3)) # (N,)
        n_obs = torch.sum(node_obs)
        pred_masked = dim_pred*obs_mask
        target_masked = dim_target*obs_mask

        se_elements = torch.nn.functional.mse_loss(pred_masked, target_masked,
                reduction="none")
        ae_elements = torch.nn.functional.l1_loss(pred_masked, target_masked,
                reduction="none")

        se_node = torch.sum(se_elements, dim=(0,1,3)) # (N,)
        ae_node = torch.sum(ae_elements, dim=(0,1,3)) # (N,)

        mse = torch.sum(se_node)/n_obs
        max_mse = torch.max(se_node/node_obs)
        mae = torch.sum(ae_node)/n_obs
        max_mae = torch.max(ae_node/node_obs)

        rmse = torch.sqrt(mse)

        dim_errors["mae"].append(mae.cpu().item())
        dim_errors["mse"].append(mse.cpu().item())
        dim_errors["rmse"].append(rmse.cpu().item())
        dim_errors["max_mse"].append(max_mse.cpu().item())
        dim_errors["max_mae"].append(max_mae.cpu().item())

    return_dict = {}
    for metric, dim_vals in dim_errors.items():
        # Metric over all output, mean over errors for y dims
        return_dict[metric] = np.mean(dim_vals)

        # Add dimension-specific metric values if d_y > 1
        if d_y > 1:
            return_dict.update({f"{metric}_d{i}": val for i, val in enumerate(dim_vals)})

    return return_dict

def build_gnn_seq(n_layers, in_dim, hidden_dim, out_dim, gnn_type):
    # Build sequential GNN model according to given description
    layer_class, use_edge_weight = constants.GNN_LAYERS[gnn_type]

    if use_edge_weight:
        gnn_signature = 'x, edge_index, edge_weight -> x'
    else:
        gnn_signature = 'x, edge_index -> x'

    layer_list = []
    if n_layers == 1:
        layer_list.append((layer_class(in_dim, out_dim), gnn_signature))
    else:
        layer_list.append((layer_class(in_dim, hidden_dim), gnn_signature))
        for layer_i in range(1, n_layers):
            layer_list.append(nn.ReLU())
            # Keep intermediate dimensions same as output dim
            layer_out_dim = out_dim if layer_i == (n_layers-1) else hidden_dim
            layer_list.append((layer_class(hidden_dim, layer_out_dim), gnn_signature))

    return ptg.nn.Sequential('x, edge_index, edge_weight', layer_list)

def forward_fill(obs, mask):
    # obs: (N_eval, N, N_T, d_y)
    # mask: (N_eval, N, N_T)

    ff_obs = torch.zeros_like(obs) # (N_eval, N, N_T, d_y)
    ff_obs[:,:,0] = obs[:,:,0] # First time step is just from obs (0 where unobserved)
    for t_i in range(1,obs.shape[2]):
        mask_slice = mask[:,:,t_i].unsqueeze(-1)
        # From obs if observed, otherswise from previous timestep
        ff_obs[:,:,t_i] = obs[:,:,t_i]*mask_slice + ff_obs[:,:,t_i-1]*(1. - mask_slice)

    return ff_obs

# Create time deltas for individual nodes
def node_t_deltas(ts, obs_mask):
    # Create time deltas
    # ts: (N_data, N_T)
    # obs_mask: (N_data, N_T, N)
    n_nodes = obs_mask.shape[2]
    repeated_t = ts.unsqueeze(2).repeat(1,1,n_nodes) # (N_samples, N_T, N)
    observed_t = repeated_t * obs_mask # (N_samples, N_T, N)
    prev_t_list = [torch.zeros(ts.shape[0], n_nodes)] # Before t_1 is just 0
    for i_t in range(1, ts.shape[1]):
        # t_slice has shape (N_samples, N)
        # In observed_t the unobserved timesteps are just 0, while observed ones are  > 0
        prev_t_list.append(torch.max(observed_t[:,:i_t,:], dim=1)[0])

    prev_t = torch.stack(prev_t_list, dim=1)
    delta_t = repeated_t - prev_t # (N_samples, N_T, N)
    return delta_t # (N_samples, N_T, N)

