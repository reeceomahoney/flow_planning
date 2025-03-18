# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flow_planning.envs  # noqa: F401,
from flow_planning.envs.maze import MazeEnv
from flow_planning.runner import ClassifierRunner, Runner
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/flow_planning", config_name="cfg.yaml")
def main(agent_cfg: DictConfig):
    # set log dir
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Logging experiment in directory: {log_dir}")

    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    # set task
    task = "Isaac-Franka-FlowPlanning"
    agent_cfg.dataset.task_name = task

    ### Create environment

    if task == "Maze":
        # create maze environment
        env = MazeEnv(agent_cfg)
        agent_cfg.obs_dim = env.obs_dim
        agent_cfg.act_dim = env.act_dim
    else:
        env_cfg = parse_env_cfg(
            task, device=agent_cfg.device, num_envs=agent_cfg.num_envs
        )
        # override config values
        env_cfg.scene.num_envs = agent_cfg.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = agent_cfg.device
        # create isaac environment
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        agent_cfg.obs_dim = 3
        agent_cfg.act_dim = 0
        env = RslRlVecEnvWrapper(env)  # type: ignore

    ### Create runner

    if task == "Isaac-Franka-FlowPlanning":
        runner = Runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif task == "Isaac-Franka-Guidance":
        runner = ClassifierRunner(
            env, agent_cfg, log_dir=log_dir, device=agent_cfg.device
        )
        model_path = "logs/diffusion/franka/Feb-28/15-30-11/" + "models/model.pt"
        runner.load(model_path)

    # dump the configuration into log-directory
    agent_cfg_dict = OmegaConf.to_container(agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg_dict)
    if task != "Maze":
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # run training
    runner.learn()

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
