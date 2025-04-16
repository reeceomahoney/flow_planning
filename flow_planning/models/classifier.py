import torch
import torch.nn as nn

from flow_planning.utils import SinusoidalPosEmb


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, timestep_dim: int, device: str):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + timestep_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        ).to(device)
        self.t_emb = SinusoidalPosEmb(timestep_dim, device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.t_emb(t.expand(x.shape[0]))
        t = t.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, t], dim=-1)
        return self.model(x)
