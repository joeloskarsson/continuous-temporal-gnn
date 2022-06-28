import torch

import utils

def train_epoch(model, data_loader, opt, loss_fn, config):
    model.train(True)

    batch_losses = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU

        cur_batch_size = batch.num_graphs
        obs_mask = batch.mask.transpose(0,1) # (N_T, B*N)
        n_obs = torch.sum(obs_mask) # number of observations in batch
        opt.zero_grad()

        # Predictions at time step i are for y at time step i (pred. using decayed state)
        full_pred = model.forward(batch) # Shape(N_T, B*N, d_y)
        target = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)

        target_loss = (target*obs_mask.unsqueeze(2))[config["init_points"]:]
        pred_loss = (full_pred*obs_mask.unsqueeze(2))[config["init_points"]:]

        # Mean over number of observatios in batch, d_y
        batch_loss = loss_fn(pred_loss, target_loss)/(n_obs*config["y_dim"])

        batch_loss.backward()
        opt.step()

        batch_losses.append(batch_loss*cur_batch_size)

    # Here mean over samples, to not weight samples in small batches higher
    epoch_loss = torch.sum(torch.stack(batch_losses))/len(data_loader.dataset)

    return epoch_loss.item()

@torch.no_grad()
def val_epoch(model, data_loader, config):
    model.train(False)

    targets = []
    predictions = []
    masks = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU

        full_pred = model.forward(batch) # Shape(N_T, B*N, 1)
        target = batch.y.transpose(0,1)[
                config["init_points"]:] # Shape (N_T-N_init, B*N, d_y)

        # Predictions at time step i are for y at time step i (pred. using decayed state)
        pred = full_pred[config["init_points"]:] # Shape (N_T-N_init, B*N, d_y)

        targets.append(target)
        predictions.append(pred)
        masks.append(batch.mask.transpose(0,1)[config["init_points"]:])

    targets_full = torch.cat(targets, dim=1) # Shape (N_T-N_init, N_data*N, d_y)
    pred_full = torch.cat(predictions, dim=1) # Shape (N_T-N_init, N_data*N, d_y)
    mask_full = torch.cat(masks, dim=1).unsqueeze(2) # (N_T-N_init, N_data*N, 1)

    return utils.eval_prediction(pred_full, targets_full, mask_full, config["num_nodes"])

