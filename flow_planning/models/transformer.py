import logging

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from flow_planning.utils import SinusoidalPosEmb

log = logging.getLogger(__name__)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        T: int,
        attn_dropout: float,
        weight_decay: float,
        device: str,
        value: bool = False,
    ):
        super().__init__()
        # variables
        input_dim = act_dim + obs_dim
        # input_len = T + 3 if value else T + 2
        input_len = T + 1
        self.weight_decay = weight_decay
        self.device = device
        self.T = T
        self.value = value

        # embeddings
        self.x_emb = nn.Linear(input_dim, d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            Rearrange("b d -> b 1 d"),
        )
        self.goal_emb = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            Rearrange("b d -> b 1 d"),
        )
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(d_model, device),
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
        )

        self.pos_emb = SinusoidalPosEmb(d_model, device)(
            torch.arange(input_len)
        ).unsqueeze(0)

        self.drop = nn.Dropout(attn_dropout)

        # transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.register_buffer("mask", self.generate_mask(input_len))
        self.ln_f = nn.LayerNorm(d_model)

        if value:
            self.output = nn.Linear(d_model, 1)
        else:
            self.output = nn.Linear(d_model, input_dim)

        self.apply(self._init_weights)
        self.to(device)

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total parameters: {total_params:e}")

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerEncoder,
            nn.ModuleList,
            nn.Sequential,
            nn.SiLU,
            DiffusionTransformer,
            SinusoidalPosEmb,
            Rearrange,
            Reduce,
            nn.Mish,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, x, t, data):
        # embed
        x_emb = self.x_emb(x)
        t_emb = self.t_emb(t).squeeze(1)
        # obs_emb = self.obs_emb(data["obs"])
        # goal_emb = self.goal_emb(data["goal"])
        # return_emb = self.return_emb(data["returns"])

        # construct input
        x = torch.cat([t_emb, x_emb], dim=1)
        x = self.drop(x + self.pos_emb)

        # output
        x = self.encoder(x)[:, -self.T :]
        x = self.ln_f(x)
        return self.output(x)

    def generate_mask(self, x):
        mask = torch.zeros(x, x, dtype=torch.bool)
        # Every token attends to the first token
        mask[:, 0] = True
        # Create indices for rows and columns
        indices = torch.arange(x)
        # Calculate absolute distance between indices
        distance = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
        # Allow attention where distance is 0 (self) or 1 (adjacent)
        mask = mask | (distance <= 1)

        # mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)

        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def detach_all(self):
        for _, param in self.named_parameters():
            param.detach_()
