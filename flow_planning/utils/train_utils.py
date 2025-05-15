import math
import os
from pathlib import Path

import gymnasium as gym
import pytorch_kinematics as pk
import torch
import torch.nn as nn

from flow_planning.envs import ParticleEnv
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def create_env(env_name, agent_cfg, video=False, resume_path=None):
    match env_name:
        case "Particle":
            env = ParticleEnv(
                num_envs=agent_cfg.num_envs,
                seed=agent_cfg.seed,
                device=agent_cfg.device,
            )
            agent_cfg.obs_dim = env.obs_dim
            agent_cfg.act_dim = env.act_dim
            env_cfg = None
        case _:
            env_cfg = parse_env_cfg(
                env_name, device=agent_cfg.device, num_envs=agent_cfg.num_envs
            )
            # override config values
            env_cfg.scene.num_envs = agent_cfg.num_envs
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = agent_cfg.device
            # create isaac environment
            env = gym.make(
                env_name, cfg=env_cfg, render_mode="rgb_array" if video else None
            )

            # video recording
            if video:
                print("[INFO] Recording video")
                assert resume_path is not None
                video_folder = Path(resume_path).parent.parent / "videos"
                os.makedirs(video_folder, exist_ok=True)

                video_kwargs = {
                    "video_folder": video_folder,
                    "step_trigger": lambda step: step == 0,
                    "video_length": env.max_episode_length,  # type: ignore
                    "disable_logger": True,
                }
                print_dict(video_kwargs, nesting=4)
                env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

            env = RslRlVecEnvWrapper(env)  # type: ignore
            agent_cfg.obs_dim = env.num_obs
            agent_cfg.act_dim = 0

    return env, agent_cfg, env_cfg


def get_goal(env, urdf_chain=None):
    if isinstance(env, RslRlVecEnvWrapper):
        assert urdf_chain is not None

        # get joint info
        init_pos = env.unwrapped.scene["robot"].data.default_joint_pos.clone()[0:1, :7]
        joint_pos_limits = env.unwrapped.scene[
            "robot"
        ].data.soft_joint_pos_limits.clone()[0]

        # get target position and orientation
        pos = torch.tensor([0.5, 0.3, 0.2], device=env.device)
        pos = env.unwrapped.command_manager.get_command("ee_pose")[0, :3]  # type: ignore
        rot = torch.tensor([0.0, math.pi, 0.0], device=env.device)
        goal_tf = pk.Transform3d(pos=pos, rot=rot, device=str(env.device))

        # solve ik
        ik = pk.PseudoInverseIK(
            urdf_chain,
            max_iterations=30,
            retry_configs=init_pos,
            joint_limits=joint_pos_limits.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2,
        )
        sol = ik.solve(goal_tf)

        # build goal state
        joint_pos = sol.solutions.squeeze(1)
        joint_vel = torch.zeros_like(joint_pos)
        goal = torch.cat([joint_pos, joint_vel], dim=-1)
    else:
        goal = env.goal
        goal = torch.tensor([0, 1, 0.6, -0.6])
    return goal


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class InferenceContext:
    """
    Context manager for inference mode
    """

    def __init__(self, runner):
        self.runner = runner
        self.policy = runner.policy
        self.ema_helper = runner.ema_helper
        self.use_ema = runner.use_ema

    def __enter__(self):
        self.inference_mode_context = torch.inference_mode()
        self.inference_mode_context.__enter__()
        self.runner.policy.eval()
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.runner.policy.train()
        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())
        self.inference_mode_context.__exit__(exc_type, exc_value, traceback)
