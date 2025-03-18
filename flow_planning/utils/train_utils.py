import math

import torch
import torch.nn as nn
from torch import Tensor


def check_collisions(traj: Tensor) -> Tensor:
    """0 if in collision, 1 otherwise"""
    x_mask = (traj[..., 0] >= 0.55) & (traj[..., 0] <= 0.65)
    y_mask = (traj[..., 1] >= -0.8) & (traj[..., 1] <= 0.8)
    z_mask = (traj[..., 2] >= 0.0) & (traj[..., 2] <= 0.6)
    return ~(x_mask & y_mask & z_mask)


def calculate_return(
    traj: Tensor, obs: Tensor, goal: Tensor, mask: Tensor, gammas: Tensor
) -> Tensor:
    obs = torch.tensor([0.4, 0.0, 0.6]).to(traj.device)
    goal = torch.tensor([0.8, 0.0, 0.6]).to(traj.device)

    rewards = torch.zeros_like(traj[..., 0])
    rewards[:, 0] = 1 - torch.exp(-torch.norm(traj[:, 0] - obs.unsqueeze(0), dim=-1))
    rewards[:, -1] = 1 - torch.exp(-torch.norm(traj[:, -1] - goal.unsqueeze(0), dim=-1))
    return rewards.unsqueeze(-1)
    # return ((reward * mask) * gammas).sum(dim=-1, keepdim=True)


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
