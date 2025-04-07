import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flow_planning.envs import MazeEnv, ParticleEnv
from flow_planning.models.unet import ConditionalUnet1D
from flow_planning.utils import Normalizer, calculate_return, get_goal
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


def expand_t(tensor: Tensor, bsz: int) -> Tensor:
    return tensor.view(1, -1, 1).expand(bsz, -1, 1)


class Policy(nn.Module):
    def __init__(
        self,
        model: ConditionalUnet1D,
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
        self.model = model
        self.env = env
        self.normalizer = normalizer
        self.device = device
        self.use_refinement = False
        self.isaac_env = isinstance(self.env, RslRlVecEnvWrapper)

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

        # training
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

        # guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)
        self.cond_mask_prob = cond_mask_prob
        self.cond_lambda = cond_lambda
        self.alpha = 0.0


        self.to(device)

    ############
    # Main API #
    ############

    @torch.no_grad()
    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data["obs"] = self.get_model_states(data["obs"])
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, self.action_dim :]
        action = x[:, : self.T_action, : self.action_dim]
        return {"action": action, "obs_traj": obs}

    def update(self, data):
        data = self.process(data)
        x_1 = data["input"]
        x_0 = torch.randn_like(x_1)
        bsz = x_1.shape[0]

        # compute sample and target
        if self.algo == "flow":
            t = torch.rand(bsz, 1, 1).to(self.device)
            x_t = (1 - t) * x_0 + t * x_1
            target = x_1 - x_0
        elif self.algo == "ddpm":
            t = torch.randint(0, self.sampling_steps, (bsz, 1, 1)).to(self.device)
            x_t = self.scheduler.add_noise(x_1, x_0, t)  # type: ignore
            target = x_0

        # cfg masking
        if self.cond_mask_prob > 0:
            cond_mask = torch.rand(bsz, 1) < self.cond_mask_prob
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
            self.scheduler.set_timesteps(self.sampling_steps, self.device)
            timesteps = self.scheduler.timesteps

        if self.cond_lambda > 0:
            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["returns"][bsz:] = 0

        # inference
        for i in range(self.sampling_steps):
            x = torch.cat([x] * 2) if self.cond_lambda > 0 else x
            x = self.inpaint(x, data)

            if self.algo == "flow":
                x = self.step(x, timesteps[i], timesteps[i + 1], data)
            elif self.algo == "ddpm":
                t = timesteps[i].view(-1, 1, 1).expand(bsz, 1, 1).float()
                out = self.model(x, t, data)
                x = self.scheduler.step(out, timesteps[i], x).prev_sample  # type: ignore

            # guidance
            if self.alpha > 0:
                grad = torch.zeros_like(x)
                grad[..., 9] = 1
                dt = timesteps[i + 1] - timesteps[i]
                x += self.alpha * (1 - timesteps[i]) * dt * grad.detach()
            elif self.cond_lambda > 0:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + self.cond_lambda * (x_cond - x_uncond)

        x = self.inpaint(x, {"obs": data["obs"][:bsz], "goal": data["goal"][:bsz]})

        # refinement step
        if self.use_refinement:
            midpoint = x[:, self.T // 2, self.action_dim :].clone()
            x_0 = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
            x = 0.5 * x_0 + 0.5 * x
            x = torch.cat([x[:, : self.T // 2], x[:, self.T // 2 :]], dim=0)

            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["obs"][bsz:] = midpoint
            data["goal"][:bsz] = midpoint[:, 18:27]

            for i in range(self.sampling_steps // 2):
                x = self.inpaint(x, data)
                x = self.step(
                    x,
                    timesteps[i + self.sampling_steps // 2],
                    timesteps[i + 1 + self.sampling_steps // 2],
                    data,
                )

            x = self.inpaint(x, data)
            x = torch.cat([x[:bsz], x[bsz:]], dim=1)

        # denormalize
        # x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_output(x)

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)

        if "action" in data:
            # train and test case
            obs = data["obs"][:, 0]
            input = torch.cat([data["action"], data["obs"]], dim=-1)
            input = self.normalizer.scale_output(input)
            returns = calculate_return(data["obs"])
            returns = self.normalizer.scale_return(returns)
        else:
            # sim case
            input = None
            obs = data["obs"]
            returns = torch.ones(obs.shape[0], 1).to(self.device)

        obs = self.normalizer.scale_input(obs)
        goal = self.normalizer.scale_input(data["goal"])

        return {"obs": obs, "input": input, "goal": goal, "returns": returns}

    def dict_to_device(self, data: dict) -> dict:
        return {k: v.to(self.device) for k, v in data.items()}

    def inpaint(self, x: Tensor, data: dict) -> Tensor:
        x[:, 0, self.action_dim :] = data["obs"]
        x[:, -1, self.action_dim :] = data["goal"]
        return x

    def get_model_states(self, x):
        return x[:, 18:27] if self.isaac_env else x

    #################
    # Visualization #
    #################

    def plot_trajectory(self, it: int):
        # get obs and goal
        obs, _ = self.env.get_observations()
        goal = get_goal(self.env)

        # plot trajectory
        if self.cond_mask_prob > 0:
            lambdas = [0, 1, 2, 5, 10]
            fig, axes = plt.subplots(1, len(lambdas), figsize=(len(lambdas) * 4, 4))

            for i in range(len(lambdas)):
                self.cond_lambda = lambdas[i]
                traj = self.act({"obs": obs, "goal": goal})["obs_traj"]
                self.generate_plot(axes[i], traj[..., 18:21], obs[:, 18:21], goal)
                axes[i].set_title(f"Lambda: {lambdas[i]}")

            self.cond_lambda = 0
            fig.tight_layout()
            wandb.log({"Guided Trajectory": wandb.Image(fig)}, step=it)
        else:
            traj = self.act({"obs": obs, "goal": goal})["obs_traj"]
            fig = plt.figure(figsize=(10, 10), dpi=300)
            ax = fig.add_subplot(projection="3d" if self.isaac_env else None)
            self.generate_plot(ax, traj, self.get_model_states(obs), goal)

            fig.tight_layout()
            wandb.log({"Trajectory": wandb.Image(fig)}, step=it)

    def generate_plot(self, ax, traj, obs, goal, color="blue", label=None):
        traj, obs, goal = traj[0].cpu(), obs[0].cpu(), goal[0].cpu()
        marker_params = {
            "markersize": 35,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 2,
            "linestyle": "None",
        }

        if self.isaac_env:
            c = torch.linspace(0, 1, len(traj)) ** 0.5
            s = [300] * len(traj)
            ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], s=s, c=c, cmap="Reds")
            ax.plot(obs[0], obs[1], obs[2], "o", **marker_params)
            ax.plot(goal[0], goal[1], goal[2], "*", **marker_params)
        else:
            c = torch.linspace(0, 1, len(traj)) ** 0.7
            ax.scatter(traj[:, 0], traj[:, 1], c=c, cmap="Reds", s=500)
            ax.plot(obs[0], obs[1], "o", **marker_params)
            ax.plot(goal[0], goal[1], "*", **marker_params)


class ClassifierPolicy(Policy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        classifier: ConditionalUnet1D,
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
        super().__init__(
            model,
            normalizer,
            env,
            obs_dim,
            act_dim,
            T,
            T_action,
            sampling_steps,
            cond_lambda,
            cond_mask_prob,
            lr,
            num_iters,
            device,
            algo,
        )
        self.classifier = classifier
        self.optimizer = AdamW(self.classifier.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

    def update(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps)
        x, t = self.truncated_forward(data, n)

        # compute model output
        pred_value = self.classifier(x, t, data)
        loss = F.mse_loss(pred_value, data["returns"])
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def test(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, n)
        pred_value = self.classifier(x, t, data)

        return F.mse_loss(pred_value, data["returns"]).item()

    @torch.no_grad()
    def forward(self, data: dict) -> torch.Tensor:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)

        if self.algo == "flow":
            timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)
        elif self.algo == "ddpm":
            self.scheduler.set_timesteps(self.sampling_steps, self.device)
            timesteps = self.scheduler.timesteps

        # inference
        for i in range(self.sampling_steps):
            x = torch.cat([x] * 2) if self.cond_lambda > 0 else x
            x = self.inpaint(x, data)

            if self.algo == "flow":
                x = self.step(x, timesteps[i], timesteps[i + 1], data)
            elif self.algo == "ddpm":
                t = timesteps[i].view(-1, 1, 1).expand(bsz, 1, 1).float()
                out = self.model(x, t, data)
                x = self.scheduler.step(out, timesteps[i], x).prev_sample  # type: ignore

            # guidance
            if self.alpha > 0:
                with torch.enable_grad():
                    x_grad = x.detach().clone().requires_grad_(True)
                    y = self.classifier(x_grad, expand_t(timesteps[i + 1], bsz), data)
                    grad = torch.autograd.grad(y.sum(), x_grad, create_graph=True)[0]
                    dt = timesteps[i + 1] - timesteps[i]
                    x += self.alpha * (1 - timesteps[i]) * dt * grad.detach()

        x = self.inpaint(x, data)

        # refinement step
        if self.use_refinement:
            midpoint = x[:, self.T // 2, self.action_dim :].clone()
            x_0 = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
            x = 0.5 * x_0 + 0.5 * x
            x = torch.cat([x[:, : self.T // 2], x[:, self.T // 2 :]], dim=0)

            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["obs"][bsz:] = midpoint
            data["goal"][:bsz] = midpoint[:, 18:27]

            for i in range(self.sampling_steps // 2):
                x = self.inpaint(x, data)
                x = self.step(
                    x,
                    timesteps[i + self.sampling_steps // 2],
                    timesteps[i + 1 + self.sampling_steps // 2],
                    data,
                )

            x = self.inpaint(x, data)
            x = torch.cat([x[:bsz], x[bsz:]], dim=1)

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
