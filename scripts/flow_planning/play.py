# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot", action="store_true", default=False, help="Whether to plot guidance."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.plot:
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import sys
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

import flow_planning.envs  # noqa: F401
import isaaclab.sim as sim_utils
from flow_planning.runner import Runner
from flow_planning.utils import create_env, get_goal, get_latest_run
from isaaclab.markers.visualization_markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)


def interpolate_color(t):
    start_color = (0.0, 0.0, 1.0)  # Blue
    end_color = (1.0, 0.0, 0.0)  # Red
    return tuple(
        start + (end - start) * t
        for start, end in zip(start_color, end_color, strict=False)
    )


def create_trajectory_visualizer(agent_cfg):
    trajectory_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Trajectory",
        markers={
            f"cuboid_{i}": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=interpolate_color(i / (agent_cfg.env.T - 1))
                ),
            )
            for i in range(agent_cfg.env.T)
        },
    )
    trajectory_visualizer = VisualizationMarkers(trajectory_visualizer_cfg)
    trajectory_visualizer.set_visibility(True)

    return trajectory_visualizer


@hydra.main(
    version_base=None, config_path="../../config/flow_planning", config_name="cfg.yaml"
)
def main(agent_cfg: DictConfig):
    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    ### Create environment
    agent_cfg.num_envs = 1
    env_name = agent_cfg.env.env_name
    env, agent_cfg, _ = create_env(env_name, agent_cfg)

    # create trajectory visualizer
    if env_name.startswith("Isaac"):
        trajectory_visualizer = create_trajectory_visualizer(agent_cfg)

    # load model runner
    runner = Runner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/flow_planning")
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.policy

    # reset environment
    obs, _ = env.get_observations()
    start = time.time()
    # simulate environment
    while simulation_app.is_running():
        goal = get_goal(env)
        output = policy.act({"obs": obs, "goal": goal})

        # plot trajectory
        if args_cli.plot:
            lambdas = torch.tensor([0, 1, 2, 3, 4])
            fig, ax = plt.subplots()
            colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(lambdas)))

            for i in range(len(lambdas)):
                policy.alpha = lambdas[i].item()
                traj = policy.act({"obs": obs, "goal": goal})["obs_traj"]
                policy.generate_plot(
                    ax,
                    traj[..., 18:21],
                    obs[:, 18:21],
                    goal,
                    color=colors[i],
                    label=f"Alpha: {lambdas[i]}",
                )

            ax.legend()
            ax.set_title("x-z plane")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.tight_layout()
            plt.show()

            simulation_app.close()
            exit()

        if env_name.startswith("Isaac"):
            trajectory_visualizer.visualize(output["obs_traj"][0, :, :3])

        # env stepping
        for i in range(runner.policy.T_action):
            obs = env.step(output["action"][:, i])[0]

            end = time.time()
            if end - start < 1 / 30:
                time.sleep(1 / 30 - (end - start))
            start = time.time()

    # close the simulator
    env.close()


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra/job_logging=disabled")
    main()
    simulation_app.close()
