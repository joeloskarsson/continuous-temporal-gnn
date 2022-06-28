import networkx as nx
import torch_geometric as ptg
import numpy as np
from matplotlib.lines import Line2D
import torch

import matplotlib.pyplot as plt

@torch.no_grad()
def plot_prediction(model, data_loader, n_plots, config):
    model.train(False)
    data_batch = next(iter(data_loader)).to(config["device"])

    pred_batch = model.forward(data_batch) # Shape(N_T, B*N, d_y)
    target_batch = data_batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
    t_batch = data_batch.t.transpose(0,1) # Shape (N_T, B)
    mask_batch = data_batch.mask.transpose(0,1) # Shape (N_T, B*N)

    pred_batch = pred_batch.reshape(
            config["time_steps"], -1, config["num_nodes"], config["y_dim"])
    target_batch = target_batch.reshape(
            config["time_steps"], -1, config["num_nodes"], config["y_dim"])
    mask_batch = mask_batch.reshape(config["time_steps"], -1, config["num_nodes"])

    # Optionally restrict number of nodes to plot for
    n_plot_nodes = min(config["num_nodes"], config["max_nodes_plot"])
    pred_batch = pred_batch[:,:,:n_plot_nodes] # (N_T, B, N_plot_nodes, d_y)
    target_batch = target_batch[:,:,:n_plot_nodes] # (N_T, B, N_plot_nodes, d_y)
    mask_batch = mask_batch[:,:,:n_plot_nodes] # (N_T, B, N_plot_nodes)

    line_colors = plt.cm.rainbow(np.linspace(0, 1, n_plot_nodes))

    figs = []
    for plot_i in range(n_plots):
        vis_target = target_batch[:,plot_i].cpu().numpy() # (N_T, N, d_y)
        vis_pred = pred_batch[:,plot_i].cpu().numpy() # (N_T, N, d_y)
        vis_mask = mask_batch[:,plot_i].cpu().numpy() # (N_T, N)
        vis_t = t_batch[:,plot_i].cpu().numpy() # (N_T,)

        fig, axes = plt.subplots(1, config["y_dim"], figsize=(7*config["y_dim"], 7),
                squeeze=False)
        # Iterate over y-dimensions
        for y_dim, ax in enumerate(axes[0]):
            for node_target, node_pred, node_mask, col in zip(vis_target[:,:,y_dim].T,
                    vis_pred[:,:,y_dim].T, vis_mask.T, line_colors):
                # Mask observations for plotted node
                node_plot_mask = (node_mask == 1)
                node_plot_times = vis_t[node_plot_mask]
                node_plot_targets = node_target[node_plot_mask]
                node_plot_pred = node_pred[node_plot_mask]

                ax.plot(node_plot_times, node_plot_targets, ls="-", marker="o",
                        zorder=2, c=col)
                ax.plot(node_plot_times, node_plot_pred, ls=":", marker="x",
                        zorder=2, c=col)

                # show warm-up area
                ax.axvspan(0., vis_t[config["init_points"]-1], color="black",
                        alpha=0.05, zorder=1)
                ax.set_xlim(vis_t[0], vis_t[-1])

        axes[0,0].legend([
                Line2D([0], [0], color="blue", ls="-", marker="o", c="blue"),
                Line2D([0], [0], color="blue", ls=":", marker="x", c="blue"),
            ], [
                "target",
                "prediction",
            ])

        figs.append(fig)

    # Return list of figures
    return figs

