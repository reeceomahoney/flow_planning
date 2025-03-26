import logging

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from flow_planning.utils import SinusoidalPosEmb

log = logging.getLogger(__name__)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        kernel_size = 5
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.conv1(x)
        embed = self.cond_encoder(cond)
        out = self.conv2(out + embed)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, obs_dim, act_dim, T, cond_dim, down_dims, device, value=False):
        super().__init__()
        input_dim = obs_dim + act_dim
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=False))

        # diffusion step embedding and observations
        self.cond_encoder = nn.Sequential(
            SinusoidalPosEmb(cond_dim, device),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.Mish(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ResBlock(dim_in, dim_out, cond_dim),
                        ResBlock(dim_out, dim_out, cond_dim),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]

        if not value:
            self.mid_modules = nn.ModuleList(
                [
                    ResBlock(mid_dim, mid_dim, cond_dim),
                    ResBlock(mid_dim, mid_dim, cond_dim),
                ]
            )

            up_modules = nn.ModuleList([])
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (len(in_out) - 1)
                up_modules.append(
                    nn.ModuleList(
                        [
                            ResBlock(dim_out * 2, dim_in, cond_dim),
                            ResBlock(dim_in, dim_in, cond_dim),
                            Upsample1d(dim_in) if not is_last else nn.Identity(),
                        ]
                    )
                )

            self.final_conv = nn.Sequential(
                ConvBlock(start_dim, start_dim),
                nn.Conv1d(start_dim, input_dim, 1),
            )
            self.up_modules = up_modules
        else:
            self.mid_block_1 = ResBlock(mid_dim, mid_dim // 2, cond_dim)
            self.mid_down_1 = Downsample1d(mid_dim // 2)
            self.mid_block_2 = ResBlock(mid_dim // 2, mid_dim // 4, cond_dim)
            self.mid_down_2 = Downsample1d(mid_dim // 4)

            fc_dim = 512
            self.final_block = nn.Sequential(
                nn.Linear(fc_dim + cond_dim, fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, 1),
            )

        self.down_modules = down_modules
        self.value = value
        self.to(device)

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total parameters: {total_params:e}")

    def forward(self, x: Tensor, t: Tensor, data_dict: dict):
        x = einops.rearrange(x, "b t h -> b h t")
        global_feature = self.cond_encoder(t.view(-1, 1)).squeeze(1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:  # type: ignore
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        if not self.value:
            for mid_module in self.mid_modules:
                x = mid_module(x, global_feature)

            for resnet, resnet2, upsample in self.up_modules:  # type: ignore
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
                x = upsample(x)

            x = self.final_conv(x)

            x = einops.rearrange(x, "b h t -> b t h")
        else:
            x = self.mid_block_1(x, global_feature)
            x = self.mid_down_1(x)
            x = self.mid_block_2(x, global_feature)
            x = self.mid_down_2(x)

            x = x.view(x.shape[0], -1)
            x = self.final_block(torch.cat([x, global_feature], dim=-1))

        return x
