import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, data_loader, device: str):
        super().__init__()
        dl = data_loader.dataset.dataset.dataset

        # scaling
        self.register_buffer("obs_max", dl.obs_max)
        self.register_buffer("obs_min", dl.obs_min)

        # limits
        limits = torch.zeros((2, self.obs_max.shape[-1]))
        self.register_buffer("limits", limits)
        self.limits[0, :] = -1 - 1e-4
        self.limits[1, :] = 1 + 1e-4

        self.to(device)

    def scale(self, x) -> torch.Tensor:
        return (x - self.obs_min) / (self.obs_max - self.obs_min) * 2 - 1

    def inverse_scale(self, x) -> torch.Tensor:
        return (x + 1) * (self.obs_max - self.obs_min) / 2 + self.obs_min

    def clip(self, y):
        return torch.clamp(y, self.limits[0, :], self.limits[1, :])
