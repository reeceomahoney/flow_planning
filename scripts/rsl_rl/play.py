# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=16, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_timesteps", type=int, default=128, help="Number of timesteps to simulate."
)
parser.add_argument(
    "--collect", action="store_true", default=False, help="Whether to collect data."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# enable headless mode for data collection
if args_cli.collect:
    args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
sys.argv.append("hydra.output_subdir=null")
sys.argv.append("hydra.run.dir=.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import hydra
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # isort: skip
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

import flow_planning.envs  # noqa: F401
from flow_planning.utils import DataCollector, get_latest_run
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


@hydra.main(
    version_base=None, config_path="../../config/rsl_rl", config_name="cfg.yaml"
)
def main(agent_cfg: DictConfig):
    # specify directory for logging experiments
    env_name = "Isaac-Franka-RL"
    resume_path = get_latest_run("logs/rsl_rl")
    env_cfg = parse_env_cfg(env_name, device=agent_cfg.device)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg.num_envs = args_cli.num_envs

    # create isaac environment
    env = gym.make(
        env_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(resume_path, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore

    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg_dict = OmegaConf.to_container(agent_cfg)
    ppo_runner = OnPolicyRunner(
        env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )

    if args_cli.collect:
        collector = DataCollector(env, "data/rsl_rl/stitch_data.hdf5")
        pbar = tqdm(total=args_cli.num_timesteps, desc="Collecting data")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    if not args_cli.collect:
        args_cli.num_timesteps = float("inf")

    # simulate environment
    while timestep < args_cli.num_timesteps:
        start = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            actions += 0.2 * torch.randn_like(actions)
            next_obs, rew, dones, _ = env.step(actions)
            # collect data
            if args_cli.collect:
                collector.add_step(obs, actions, rew, dones)
            obs = next_obs

        end = time.time()
        if not args_cli.collect and end - start < 1 / 30:
            time.sleep(1 / 30 - (end - start))

        timestep += 1
        if args_cli.collect:
            pbar.update(1)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    collector.flush()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()
