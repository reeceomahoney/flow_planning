import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def plot_maze(maze: torch.Tensor, figsize: tuple = (8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(maze, cmap="gray", extent=(-4, 4, -4, 4))
    return fig, ax


def plot_obstacle(ax, obstacle: torch.Tensor):
    obstacle_square = Rectangle(
        (obstacle[0, 0].item(), obstacle[0, 1].item()),
        1.0,
        1.0,
        facecolor="red",
        alpha=0.5,
    )
    ax.add_patch(obstacle_square)


def plot_3d_guided_trajectory(
    policy,
    obs: torch.Tensor,
    goal: torch.Tensor,
    obstacle: torch.Tensor,
    scales: list,
    alphas_or_lambdas: str,
):
    fig, axes = plt.subplots(
        1, len(scales), figsize=(16, 3.5), subplot_kw={"projection": "3d"}
    )
    goal_ = goal.cpu().numpy()

    # obstacle
    cuboid_vertices = [
        [0.55, -0.8, 0.0],
        [0.65, -0.8, 0.0],
        [0.65, 0.8, 0.0],
        [0.55, 0.8, 0.0],
        [0.55, -0.8, 0.6],
        [0.65, -0.8, 0.6],
        [0.65, 0.8, 0.6],
        [0.55, 0.8, 0.6],
    ]
    cuboid_faces = [
        [cuboid_vertices[j] for j in [0, 1, 5, 4]],
        [cuboid_vertices[j] for j in [1, 2, 6, 5]],
        [cuboid_vertices[j] for j in [2, 3, 7, 6]],
        [cuboid_vertices[j] for j in [3, 0, 4, 7]],
        [cuboid_vertices[j] for j in [0, 1, 2, 3]],
        [cuboid_vertices[j] for j in [4, 5, 6, 7]],
    ]

    for i, scale in enumerate(scales):
        # set guidance scale
        if alphas_or_lambdas == "alphas":
            policy.alpha = scale
        elif alphas_or_lambdas == "lambdas":
            policy.cond_lambda = scale
        else:
            raise ValueError(
                f"Invalid argument: {alphas_or_lambdas}. Must be 'alphas' or 'lambdas'."
            )

        # Compute trajectory
        traj = policy.act({"obs": obs, "obstacle": obstacle, "goal": goal})
        traj = traj["obs_traj"][0, :, 18:21].detach().cpu().numpy()

        # Plot trajectory with color gradient
        gradient = np.linspace(0, 1, len(traj))
        axes[i].scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=gradient, cmap="inferno")

        # Plot start and goal positions
        marker_params = {"markersize": 10, "markeredgewidth": 3}
        axes[i].plot(
            traj[0, 0], traj[0, 1], traj[0, 2], "x", color="green", **marker_params
        )
        axes[i].plot(
            goal_[0, 0], goal_[0, 1], goal_[0, 2], "x", color="red", **marker_params
        )
        # axes[i].add_collection3d(
        #     Poly3DCollection(cuboid_faces, alpha=0.5, facecolor="red")
        # )
        axes[i].view_init(elev=0, azim=90)
        axes[i].set_title(f"{alphas_or_lambdas}={scale}")

    # reset guidance scale
    if alphas_or_lambdas == "alphas":
        policy.alpha = 0
    elif alphas_or_lambdas == "lambdas":
        policy.cond_lambda = 0
    else:
        raise ValueError(
            f"Invalid argument: {alphas_or_lambdas}. Must be 'alphas' or 'lambdas'."
        )

    fig.tight_layout()
    return fig
