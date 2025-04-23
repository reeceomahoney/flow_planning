import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, data_loader, scaling: str, device: str):
        super().__init__()
        dl = data_loader.dataset.dataset.dataset

        # linear scaling
        self.register_buffer("x_max", dl.x_max)
        self.register_buffer("x_min", dl.x_min)
        self.register_buffer("y_max", dl.y_max)
        self.register_buffer("y_min", dl.y_min)
        self.register_buffer("r_max", dl.r_max)
        self.register_buffer("r_min", dl.r_min)
        self.register_buffer("act_max", dl.act_max)
        self.register_buffer("act_min", dl.act_min)

        # gaussian scaling
        self.register_buffer("x_mean", dl.x_mean)
        self.register_buffer("x_std", dl.x_std)
        self.register_buffer("y_mean", dl.y_mean)
        self.register_buffer("y_std", dl.y_std)

        # bounds
        y_bounds = torch.zeros((2, self.y_mean.shape[-1]))
        self.register_buffer("y_bounds", y_bounds)
        if scaling == "linear":
            self.y_bounds[0, :] = -1 - 1e-4
            self.y_bounds[1, :] = 1 + 1e-4
        elif scaling == "gaussian":
            self.y_bounds[0, :] = -5
            self.y_bounds[1, :] = 5

        self.scaling = scaling
        self.to(device)

    def scale_input(self, x) -> torch.Tensor:
        if self.scaling == "linear":
            return (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        elif self.scaling == "gaussian":
            return (x - self.x_mean) / self.x_std
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_goal(self, x) -> torch.Tensor:
        if self.scaling == "linear":
            return (x - self.x_min[:2]) / (self.x_max[:2] - self.x_min[:2]) * 2 - 1
        elif self.scaling == "gaussian":
            return (x - self.x_mean[:2]) / self.x_std[:2]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_3d_pos(self, pos) -> torch.Tensor:
        if self.scaling == "linear":
            return (pos - self.x_min[18:21]) / (
                self.x_max[18:21] - self.x_min[18:21]
            ) * 2 - 1
        elif self.scaling == "gaussian":
            return (pos - self.x_mean[18:21]) / self.x_std[18:21]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_9d_pos(self, pos) -> torch.Tensor:
        if self.scaling == "linear":
            return (pos - self.x_min[18:27]) / (
                self.x_max[18:27] - self.x_min[18:27]
            ) * 2 - 1
        elif self.scaling == "gaussian":
            return (pos - self.x_mean[18:27]) / self.x_std[18:27]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_output(self, y) -> torch.Tensor:
        if self.scaling == "linear":
            return (y - self.y_min) / (self.y_max - self.y_min) * 2 - 1
        elif self.scaling == "gaussian":
            return (y - self.y_mean) / self.y_std
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_action(self, a) -> torch.Tensor:
        return (a - self.act_min) / (self.act_max - self.act_min) * 2 - 1

    def inverse_scale_action(self, a) -> torch.Tensor:
        return (a + 1) * (self.act_max - self.act_min) / 2 + self.act_min

    def inverse_scale_output(self, y) -> torch.Tensor:
        if self.scaling == "linear":
            return (y + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        elif self.scaling == "gaussian":
            return y * self.y_std + self.y_mean
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_return(self, r) -> torch.Tensor:
        return (r - self.r_min) / (self.r_max - self.r_min)

    def clip(self, y):
        return torch.clamp(y, self.y_bounds[0, :], self.y_bounds[1, :])
