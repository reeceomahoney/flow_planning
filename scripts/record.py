"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_timesteps", type=int, default=128, help="Number of timesteps to simulate."
)
parser.add_argument(
    "--collect", action="store_true", default=False, help="Whether to collect data."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.collect:
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import flow_planning.envs  # noqa: F401
from flow_planning.utils.data_collector import DataCollector
from flow_planning.utils.train_utils import create_env
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


@hydra.main(
    version_base=None, config_path="../../config/flow_planning", config_name="cfg.yaml"
)
def main(agent_cfg: DictConfig):
    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    # create environment
    agent_cfg.num_envs = args_cli.num_envs
    env_name = "Isaac-Franka-IK"
    env, agent_cfg, _ = create_env(env_name, agent_cfg)
    assert isinstance(env, RslRlVecEnvWrapper)
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    arm_action = env.unwrapped.action_manager.get_term("arm_action")

    # create data collector
    if args_cli.collect:
        collector = DataCollector(env, "data/ik/data.hdf5")
        pbar = tqdm(total=args_cli.num_timesteps, desc="Collecting data")
        max_vals = torch.tensor([50, 50, 50, 50, 3, 3, 3]).to(device=env.device)
    else:
        args_cli.num_timesteps = float("inf")

    obs, _ = env.get_observations()
    timestep = 0
    while timestep < args_cli.num_timesteps:
        start = time.time()

        # step
        goal = env.unwrapped.command_manager.get_command("ee_pose")
        next_obs, rew, dones, _ = env.step(goal)

        # collect data
        if args_cli.collect and timestep % 2 == 0:
            action = torch.clamp(arm_action._final_action, -max_vals, max_vals)  # type: ignore
            collector.add_step(obs, action, rew, dones)

        obs = next_obs

        end = time.time()
        if not args_cli.collect and end - start < 1 / 30:
            time.sleep(1 / 30 - (end - start))

        timestep += 1
        if args_cli.collect:
            pbar.update(1)

    collector.flush()
    env.close()


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra/job_logging=disabled")
    main()
    simulation_app.close()
