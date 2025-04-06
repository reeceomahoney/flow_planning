from isaaclab.app import AppLauncher

# NOTE: We need to run the app launcher first to avoid import errors
app_launcher = AppLauncher({"headless": True})

import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from omegaconf import DictConfig

from flow_planning.utils.dataset import get_dataloaders


def plot_lines(traj, c, ax, cmap):
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,  # type: ignore
        cmap=cmap,
        linewidth=5,
        capstyle="round",
    )
    lc.set_array(c)
    ax.add_collection(lc)


@hydra.main(
    version_base=None, config_path="../../config/flow_planning", config_name="cfg.yaml"
)
def main(agent_cfg: DictConfig):
    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    train_loader, _ = get_dataloaders(**agent_cfg.dataset)
    batch = next(iter(train_loader))
    bsz = 20
    obs = batch["obs"][:bsz]

    _, ax = plt.subplots(figsize=(10, 10), dpi=300)
    c = np.linspace(0, 1, obs.shape[1]) ** 0.7

    traj_1, traj_2 = [], []
    for i in range(obs.shape[0]):
        dist = torch.norm(obs[i, 0])
        if dist < 0.2:
            traj_1.append(obs[i, :, :2])
        else:
            traj_2.append(obs[i, :, :2])

    for traj in traj_1:
        plot_lines(traj, c, ax, "Blues")

    for traj in traj_2:
        plot_lines(traj, c, ax, "Reds")

    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Visualization of Trajectories by Starting Position")
    plt.axis("equal")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)

    save_path = "data.png"
    plt.savefig(save_path)
    print(f"Saving dataset visualization to {save_path}...")


if __name__ == "__main__":
    main()
