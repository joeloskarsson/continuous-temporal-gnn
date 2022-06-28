import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Latex styling
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

obs = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 0, 1],
])
w = np.array([
    [1, 0.3, 1, 1],
    [1, 1, 1, 0.1],
    [0.5, 0.6, 1, 1],
    [1, 1, 1, 1],
])
h_hat = np.array([
    [1, 1, 1, 1],
    [0.8, 1, 1, 0.4],
    [0.5, 0.1, 1, 0.4],
    [0.8, 1, 1, 0.7],
])
h_bar = np.array([
    [0.5, 0.1, 0, 0],
    [0.2, 0, 0.7, 0],
    [0.8, 1, 0, 0.],
    [0.2, 0, 0, 0.7],
])

T = 5
ts = np.array([0, 1, 1.7, 4])

fig, axes = plt.subplots(4,1, figsize=(4,4))

for ax, node_obs, node_w, node_h_hat, node_h_bar in zip(
        axes, obs, w, h_hat, h_bar):
    obs_mask = node_obs.astype(bool)
    node_w = node_w[obs_mask]
    node_h_hat = node_h_hat[obs_mask]
    node_h_bar = node_h_bar[obs_mask]
    node_t = ts[obs_mask]

    plot_times = np.linspace(0., T, 1000)
    last_t = 0.
    hs = []
    for t in plot_times:
        if node_t.size > 0 and t >= node_t[0]:
            cur_h_hat = node_h_hat[0]
            cur_h_bar = node_h_bar[0]
            cur_w = node_w[0]
            last_t = node_t[0]

            node_t = node_t[1:]
            node_h_hat = node_h_hat[1:]
            node_h_bar = node_h_bar[1:]
            node_w = node_w[1:]

        dt = t - last_t
        decay = np.exp(-cur_w*dt)
        h_t = cur_h_hat*decay + cur_h_bar*(1-decay)
        hs.append(h_t)

    hs = np.array(hs)

    ax.plot(plot_times, hs, linewidth=1, c="green")

    ax.set_ylim(0., 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("obs_illustration.pdf", bbox_inches = 'tight', pad_inches = 0)

