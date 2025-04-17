import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, device: str):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        ).to(device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1, 1).expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, t], dim=-1)
        return self.model(x)
