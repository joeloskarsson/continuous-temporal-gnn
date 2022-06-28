import numpy as np
import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt

# Latex styling
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

res_dir = "traffic_res"
model_files = {
    "Predict Prev.": ["traffic_test_previous.csv"],
    "GRU-D-Joint": ["traffic_test_gru.csv"],
    "GRU-D-Node": ["traffic_test_node_la.csv", "traffic_test_node_bay_PEMS04.csv"],
    "GRU-D-Graph": ["traffic_test_graph_bay.csv", "traffic_test_graph_la_PEMS04.csv"],
}

metrics = ["test_rmse",]
data_spec = {
    "bay": [0.1, 0.25, 0.5, 0.75, 1.0],
    "la": [0.1, 0.25, 0.5, 0.75, 1.0],
}
ds_ylims = {
    "bay": (0.31, 1.13),
    "la": (0.27, 0.67),
}
ds_ytickfreq = {
    "bay": 0.1,
    "la": 0.05,
}

colors = ("purple", "limegreen", "red", "orange")
line_styles = (":", (4, (1,2,7,2)), "--", "-")
decay_type = "dynamic"
marker = "."
linewidth = 1.0
markersize = 6

# Concatenate neccesary files
model_dfs = {model_name: pd.concat([pd.read_csv(os.path.join(res_dir, filename))
        for filename in model_files]).sort_values(by=["dataset"])
    for model_name, model_files in model_files.items()}

for ds, obs_fractions in data_spec.items():
    for metric in metrics:
        dataset_names = [f"{ds}_node_{frac}" for frac in obs_fractions]

        # Plot
        fig, ax = plt.subplots(figsize=(3.3,2.6))
        for (model, df), col, ls in zip(model_dfs.items(), colors, line_styles):
            if "decay_type" in df:
                # Models with different decay
                dt_df = df[df["decay_type"] == decay_type]
                dc_vals = np.stack([
                        dt_df[dt_df["dataset"] == ds_name][metric].to_numpy()
                    for ds_name in dataset_names], axis=0)

                metric_means = np.mean(dc_vals, axis=1)
                metric_stds = np.std(dc_vals, axis=1, ddof=1)

                # Plot confidence interval
                ax.fill_between(obs_fractions, metric_means-1.96*metric_stds,
                        metric_means+1.96*metric_stds, color=col, alpha=0.25, lw=0)
            else:
                # Baselines
                ds_vals = np.concatenate([df[df["dataset"] == ds_name][metric].to_numpy()
                         for ds_name in dataset_names])
                metric_means = ds_vals

            # Plot mean line
            ax.plot(obs_fractions, metric_means, c=col, ls=ls, label=f"{model}",
                    marker=marker, lw=linewidth, markersize=markersize)

        ax.legend(handlelength=3, loc="upper right")

        ax.set_ylabel("RMSE")

        ax.set_ylim(*ds_ylims[ds])
        # Set frequency of yticks
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ds_ytickfreq[ds]))

        ax.set_xlim(0.1, 1.0)
        ax.set_xlabel("Observed fraction")
        ax.set_xticks(obs_fractions)
        ax.set_xticklabels([f"{int(100*frac)}\%" for frac in obs_fractions])

        plt.tight_layout()
        plt.savefig(f"{ds}_{metric}.pdf", bbox_inches = 'tight', pad_inches = 0)

