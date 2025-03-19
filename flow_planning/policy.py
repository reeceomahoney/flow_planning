import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flow_planning.envs import MazeEnv, ParticleEnv
from flow_planning.models.transformer import DiffusionTransformer
from flow_planning.utils import Normalizer
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


def expand_t(tensor: Tensor, bsz: int) -> Tensor:
    return tensor.view(1, -1, 1).expand(bsz, -1, 1)


def to_np(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class Policy(nn.Module):
    def __init__(
        self,
        model: DiffusionTransformer,
        classifier: DiffusionTransformer | None,
        normalizer: Normalizer,
        env: RslRlVecEnvWrapper | ParticleEnv | MazeEnv,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_action: int,
        sampling_steps: int,
        cond_lambda: int,
        cond_mask_prob: float,
        lr: float,
        num_iters: int,
        device: str,
        algo: str,
    ):
        super().__init__()
        # model
        if classifier is not None:
            self.classifier = classifier
            self.alpha = 0.0
        self.model = model
        # other classes
        self.env = env
        self.normalizer = normalizer
        self.device = device

        # dims
        self.input_dim = act_dim + obs_dim
        self.action_dim = act_dim
        self.T = T
        self.T_action = T_action

        # diffusion / flow matching
        self.sampling_steps = sampling_steps
        self.beta_dist = torch.distributions.beta.Beta(1.5, 1.0)
        self.scheduler = DDPMScheduler(self.sampling_steps)
        self.algo = algo  # ddpm or flow

        # optimizer and lr scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.classifier_optimizer = AdamW(self.classifier.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.classifier_lr_scheduler = CosineAnnealingLR(
            self.classifier_optimizer, T_max=num_iters
        )

        # guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)
        self.cond_mask_prob = cond_mask_prob
        self.cond_lambda = cond_lambda

        self.to(device)

    ############
    # Main API #
    ############

    @torch.no_grad()
    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, self.action_dim :]
        # action = x[:, : self.T_action, : self.action_dim]
        action = torch.zeros_like(obs[..., :2])
        return {"action": action, "obs_traj": obs}

    def update(self, data):
        data = self.process(data)

        # sample noise and timestep
        x_1 = data["input"]
        x_0 = torch.randn_like(x_1)

        if self.algo == "flow":
            # samples = self.beta_dist.sample((len(x_1), 1, 1)).to(self.device)
            # t = 0.999 * (1 - samples)
            t = torch.rand(x_1.shape[0], 1, 1).to(self.device)
            # compute target
            x_t = (1 - t) * x_0 + t * x_1
            target = x_1 - x_0

        elif self.algo == "ddpm":
            t = torch.randint(0, self.sampling_steps, (x_1.shape[0], 1)).to(self.device)
            x_t = self.scheduler.add_noise(x_1, x_0, t)  # type: ignore
            target = x_0

        # cfg masking
        if self.cond_mask_prob > 0:
            cond_mask = torch.rand(x_1.shape[0], 1) < self.cond_mask_prob
            data["returns"][cond_mask] = 0

        # compute model output
        x_t = self.inpaint(x_t, data)
        out = self.model(x_t, t.float(), data)
        out = self.inpaint(out, data)

        loss = F.mse_loss(out, target)
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    @torch.no_grad()
    def test(self, data: dict) -> tuple[float, float, float]:
        data = self.process(data)
        x = self.forward(data)

        # calculate losses
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = F.mse_loss(x, input, reduction="none")
        obs_loss = loss[:, :, self.action_dim :].mean().item()
        action_loss = loss[:, :, : self.action_dim].mean().item()

        return loss.mean().item(), obs_loss, action_loss

    ##################
    # Classifier API #
    ##################

    def update_classifier(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps)
        x, t = self.truncated_forward(data, n)

        # compute model output
        pred_value = self.classifier(x, t, data)
        loss = F.mse_loss(pred_value, data["returns"])
        # update model
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        self.classifier_lr_scheduler.step()

        return loss.item()

    def test_classifier(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, n)
        pred_value = self.classifier(x, t, data)

        return F.mse_loss(pred_value, data["returns"]).item()

    #####################
    # Inference backend #
    #####################

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, data: dict) -> Tensor:
        t_start = expand_t(t_start, x_t.shape[0])
        t_end = expand_t(t_end, x_t.shape[0])
        return x_t + (t_end - t_start) * self.model(x_t, t_start, data)

    @torch.no_grad()
    def forward(self, data: dict) -> torch.Tensor:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)

        if self.algo == "flow":
            timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)
        elif self.algo == "ddpm":
            self.scheduler.set_timesteps(self.sampling_steps)
            timesteps = self.scheduler.timesteps

        if self.cond_lambda > 0:
            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["returns"][bsz:] = 0

        x = self.inpaint(x, data)

        # inference
        for i in range(self.sampling_steps):
            x = torch.cat([x] * 2) if self.cond_lambda > 0 else x
            if self.algo == "flow":
                x = self.step(x, timesteps[i], timesteps[i + 1], data)
            elif self.algo == "ddpm":
                t = timesteps[i].view(-1, 1).expand(bsz, 1).float()
                out = self.model(x, t, data)
                x = self.scheduler.step(out, timesteps[i], x).prev_sample  # type: ignore

            # guidance
            if self.alpha > 0:
                with torch.enable_grad():
                    x_grad = x.detach().clone().requires_grad_(True)
                    y = self.classifier(x_grad, expand_t(timesteps[i + 1], bsz), data)
                    grad = torch.autograd.grad(y.sum(), x_grad, create_graph=True)[0]
                    x = x_grad + self.alpha * (1 - timesteps[i + 1]) * grad.detach()
            elif self.cond_lambda > 0:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + self.cond_lambda * (x_cond - x_uncond)

        x = self.inpaint(x, {"obs": data["obs"][:bsz], "goal": data["goal"][:bsz]})

        # denormalize
        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_output(x)

    @torch.no_grad()
    def truncated_forward(
        self, data: dict, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
        time_steps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)
        time_steps = time_steps[: n + 1]

        # inference
        # TODO: change this to batch samples from every step
        for i in range(n):
            x = self.step(x, time_steps[i], time_steps[i + 1], data)

        return x, expand_t(time_steps[-1], bsz)

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)

        if "action" in data:
            # train and test case
            obs = data["obs"]
            input = self.normalizer.scale_output(obs)
            goal = self.normalizer.scale_goal(data["goal"][:, 0, :2])
        else:
            # sim case
            input = None
            obs = data["obs"].unsqueeze(1)
            goal = self.normalizer.scale_goal(data["goal"])

        obs = self.normalizer.scale_input(obs[:, 0])
        return {"obs": obs, "input": input, "goal": goal}

    ###########
    # Helpers #
    ###########

    def dict_to_device(self, data: dict) -> dict:
        return {k: v.to(self.device) for k, v in data.items()}

    def inpaint(self, x: Tensor, data: dict) -> Tensor:
        x[:, 0, self.action_dim : self.action_dim + 2] = data["obs"][:, :2]
        x[:, -1, self.action_dim : self.action_dim + 2] = data["goal"]
        return x

    def plot_trajectory(self, it: int):
        # get obs and goal
        if isinstance(self.env, RslRlVecEnvWrapper):
            obs, _ = self.env.get_observations()
            obs = obs[0:1, 18:21]
            goal = self.env.unwrapped.command_manager.get_command("ee_pose")  # type: ignore
            goal = goal[0:1, :3]
            # rot_mat = matrix_from_quat(goal[:, 3:])
            # ortho6d = rot_mat[..., :2].reshape(-1, 6)
            # goal = torch.cat([goal[:, :3], ortho6d], dim=-1)[0].unsqueeze(0)
        else:
            obs = torch.zeros((1, 2), device=self.device)
            goal = torch.zeros((1, 2), device=self.device)
            goal[0, 0] = 1.0

        # plot trajectory
        if self.cond_lambda > 0:
            lambdas = [0, 1, 2, 5, 10]
            fig, axes = plt.subplots(1, len(lambdas), figsize=(len(lambdas) * 4, 4))

            for i in range(len(lambdas)):
                self.cond_lambda = lambdas[i]
                traj = self.act({"obs": obs, "goal": goal})["obs_traj"]
                # traj = torch.cat([traj[0, :, 0:1], traj[0, :, 2:3]], dim=-1)
                self._generate_plot(axes[i], traj[0], obs[0], goal[0])

            self.cond_lambda = 0
        else:
            traj = self.act({"obs": obs, "goal": goal})["obs_traj"]
            # traj = torch.cat([traj[0, :, 0:1], traj[0, :, 2:3]], dim=-1)
            fig, ax = plt.subplots()
            self._generate_plot(ax, traj[0], obs[0], goal[0])

        fig.tight_layout()
        wandb.log({"Trajectory": wandb.Image(fig)}, step=it)

    def _generate_plot(self, ax, traj, obs, goal):
        traj, obs, goal = to_np(traj), to_np(obs), to_np(goal)
        marker_params = {"markersize": 10, "markeredgewidth": 3}
        # Plot trajectory with color gradient
        gradient = np.linspace(0, 1, len(traj))
        ax.scatter(traj[:, 0], traj[:, 1], c=gradient, cmap="inferno")
        # Plot start and goal positions
        ax.plot(obs[0], obs[1], "x", color="green", **marker_params)
        ax.plot(goal[0], goal[1], "x", color="red", **marker_params)
