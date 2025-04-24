import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, data_loader, device: str):
        super().__init__()
        dl = data_loader.dataset.dataset.dataset

        # scaling
        self.register_buffer("obs_max", dl.obs_max)
        self.register_buffer("obs_min", dl.obs_min)
        self.register_buffer("r_max", dl.r_max)
        self.register_buffer("r_min", dl.r_min)

        # bounds
        y_bounds = torch.zeros((2, self.y_mean.shape[-1]))
        self.register_buffer("y_bounds", y_bounds)
        self.y_bounds[0, :] = -1 - 1e-4
        self.y_bounds[1, :] = 1 + 1e-4

        self.to(device)

    def scale_obs(self, x) -> torch.Tensor:
        return (x - self.obs_min) / (self.obs_max - self.obs_min) * 2 - 1

    def scale_goal(self, x) -> torch.Tensor:
        return (x - self.obs_min[18:27]) / (
            self.obs_max[18:27] - self.obs_min[18:27]
        ) * 2 - 1

    def scale_return(self, r) -> torch.Tensor:
        return (r - self.r_min) / (self.r_max - self.r_min)

    def inverse_scale_obs(self, x) -> torch.Tensor:
        return (x + 1) * (self.obs_max - self.obs_min) / 2 + self.obs_min

    def clip(self, y):
        return torch.clamp(y, self.y_bounds[0, :], self.y_bounds[1, :])
