import torch
import argparse
import numpy as np

import utils

parser = argparse.ArgumentParser(description='Evaluate toy baselines')

parser.add_argument("--baseline", type=str, default="mean",
        help="Which baseline to use")
parser.add_argument("--dataset", type=str, default="sin_10_100",
        help="Which dataset to use")
parser.add_argument("--test", type=int, default=0,
        help="Evaluate test set (otherwise validation set)")
parser.add_argument("--init_points", type=int, default=5,
        help="Number of points to observe before prediction start")
parser.add_argument("--seed", type=int, default=42,
        help="Seed for random number generator")
config = vars(parser.parse_args())

# Set all random seeds
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# Load data
device = torch.device("cpu")
data_dict = utils.load_data(config["dataset"])
train_y = utils.to_tensor(data_dict["train"]["y"],
        device=device) # Shape (N_train, N_T, N, d_y)
train_t = utils.to_tensor(data_dict["train"]["t"], device=device) # (N_train, N_T)
train_delta_t = utils.to_tensor(data_dict["train"]["delta_t"],
        device=device) # (N_train, N, N_T)
train_mask = utils.to_tensor(data_dict["train"]["mask"],
        device=device) # (N_train, N, N_T)
edge_index = utils.to_tensor(data_dict["edge_index"], device=device, dtype=torch.long)

eval_set = "test" if config["test"] else "val"
eval_y = utils.to_tensor(data_dict[eval_set]["y"],
        device=device).transpose(1,2) # Shape (N_eval, N, N_T, d_y)
eval_t = utils.to_tensor(data_dict[eval_set]["t"], device=device) # (N_eval, N_T)
eval_delta_t = utils.to_tensor(data_dict[eval_set]["delta_t"],
        device=device) # (N_eval, N, N_T)
eval_mask = utils.to_tensor(data_dict[eval_set]["mask"],
        device=device) # (N_eval, N, N_T)
target = eval_y[:, :, config["init_points"]:]
num_nodes = eval_y.shape[1]

dataset_config = utils.load_config(config["dataset"])

# Baselines
if config["baseline"] == "mean":
    # Mean prediction
    y_mean = torch.mean(train_y)
    prediction = y_mean*torch.ones_like(target)

elif config["baseline"] == "previous":
    # Predict value at the previous observed time-step
    ff_obs = utils.forward_fill(eval_y, eval_mask)
    prediction = ff_obs[:, :, config["init_points"]-1:-1]
else:
    assert False, f"Unknown baseline: {config['baseline']}"

metric_mask = eval_mask[:, :, config["init_points"]:].unsqueeze(-1)
# Reshape prediction, target, mask
# Shape (N_eval, N, N_T, d_y)
prediction = prediction.flatten(0,1).transpose(0,1) # (N_T, N_eval*N, d_y)
target = target.flatten(0,1).transpose(0,1) # (N_T, N_eval*N, d_y)
metric_mask = metric_mask.flatten(0,1).transpose(0,1) # (N_T, N_eval*N, 1)

metrics = utils.eval_prediction(prediction, target, metric_mask, num_nodes)

print(f"model: {config['baseline']}")
print(f"dataset: {config['dataset']}")
formatted_vals = []
for name, val in metrics.items():
    print(f"{eval_set}_{name}: {val}")

