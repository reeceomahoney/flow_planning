import math

import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor

from flow_planning.envs import ParticleEnv
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def check_collisions(traj: Tensor) -> Tensor:
    """0 if in collision, 1 otherwise"""
    x_mask = (traj[..., 0] >= 0.3) & (traj[..., 0] <= 0.7)
    y_mask = (traj[..., 1] >= -0.1) & (traj[..., 1] <= 0.1)
    z_mask = (traj[..., 2] >= 0.0) & (traj[..., 2] <= 0.5)
    return ~(x_mask & y_mask & z_mask)


def calculate_return(obs: Tensor) -> Tensor:
    collision_mask = check_collisions(obs[..., 18:21]).float()
    return collision_mask.unsqueeze(-1)


def create_env(env_name, agent_cfg):
    match env_name:
        case "Particle":
            env = ParticleEnv(
                num_envs=agent_cfg.num_envs,
                seed=agent_cfg.seed,
                device=agent_cfg.device,
            )
            agent_cfg.obs_dim = env.obs_dim
            agent_cfg.act_dim = 0
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
            env = gym.make(env_name, cfg=env_cfg, render_mode=None)
            env = RslRlVecEnvWrapper(env)  # type: ignore
            agent_cfg.obs_dim = env.num_obs
            agent_cfg.act_dim = 0

    return env, agent_cfg, env_cfg


def get_goal(env):
    if isinstance(env, RslRlVecEnvWrapper):
        # (0.5, 0.3, 0.2)
        joint_pos = torch.tensor([1.2836e-01, 2.8420e-01, 4.3287e-01, -2.0772e00, -1.6051e-01, 2.3218e00, 1.4306e00])  # fmt: off
        joint_pos = joint_pos.expand(env.num_envs, -1).to(env.device)
        joint_vel = torch.zeros_like(joint_pos)
        goal = torch.cat([joint_pos, joint_vel], dim=-1)
    else:
        # goal = env.goal
        goal = torch.tensor([[1, 0, 0.33, -0.33]])
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
