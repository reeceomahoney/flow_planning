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
from flow_planning.models.transformer import DiffusionTransformer
from flow_planning.utils import Normalizer, calculate_return, get_goal
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


def expand_t(tensor: Tensor, bsz: int) -> Tensor:
    return tensor.view(1, -1, 1).expand(bsz, -1, 1)


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
        self.use_refinement = False

        self.to(device)

    ############
    # Main API #
    ############

    @torch.no_grad()
    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data["obs"] = data["obs"][:, 18:27]
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, self.action_dim :]
        # action = x[:, : self.T_action, : self.action_dim]
        action = torch.zeros(x.shape[0], x.shape[1], 7).to(self.device)
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
                with torch.enable_grad():
                    # x_grad = x.detach().clone().requires_grad_(True)
                    # y = self.classifier(x_grad, expand_t(timesteps[i + 1], bsz), data)
                    # grad = torch.autograd.grad(y.sum(), x_grad, create_graph=True)[0]
                    grad = torch.zeros_like(x)
                    grad[..., 2] = 1
                    dt = timesteps[i + 1] - timesteps[i]
                    x += self.alpha * (1 - timesteps[i]) * dt * grad.detach()
            elif self.cond_lambda > 0:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + self.cond_lambda * (x_cond - x_uncond)

        x = self.inpaint(x, {"obs": data["obs"][:bsz], "goal": data["goal"][:bsz]})

        # refinement step
        if self.use_refinement:
            midpoint = x[:, self.T // 2]
            x = torch.randn((2 * bsz, self.T // 2, self.input_dim)).to(self.device)
            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["obs"][bsz:] = midpoint
            data["goal"][:bsz] = midpoint

            for i in range(self.sampling_steps):
                x = self.inpaint(x, data)
                x = self.step(x, timesteps[i], timesteps[i + 1], data)

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

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)

        if "action" in data:
            # train and test case
            obs = data["obs"][:, 0]
            # input = torch.cat([data["action"], data["obs"]], dim=-1)
            input = data["obs"]
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

    ###########
    # Helpers #
    ###########

    def dict_to_device(self, data: dict) -> dict:
        return {k: v.to(self.device) for k, v in data.items()}

    def inpaint(self, x: Tensor, data: dict) -> Tensor:
        x[:, 0, self.action_dim :] = data["obs"]
        x[:, -1, self.action_dim :] = data["goal"]
        return x

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
                self.generate_plot(axes[i], traj[0], obs[0, 18:21], goal[0])
                axes[i].set_title(f"Lambda: {lambdas[i]}")

            self.cond_lambda = 0
            fig.tight_layout()
            wandb.log({"Guided Trajectory": wandb.Image(fig)}, step=it)
        else:
            traj = self.act({"obs": obs, "goal": goal})["obs_traj"]
            fig, ax = plt.subplots()
            self.generate_plot(ax, traj[0], obs[0, 18:21], goal[0])

            fig.tight_layout()
            wandb.log({"Trajectory": wandb.Image(fig)}, step=it)

    def generate_plot(self, ax, traj, obs, goal, color="blue", label=None):
        traj, obs, goal = traj.cpu(), obs.cpu(), goal.cpu()
        idx = 2 if isinstance(self.env, RslRlVecEnvWrapper) else 1
        marker_params = {"markersize": 10, "markeredgewidth": 3}

        # Plot trajectory with color gradient
        # gradient = np.linspace(0, 1, len(traj))
        ax.scatter(traj[:, 0], traj[:, idx], color=color, label=label)
        # Plot start and goal positions
        ax.plot(obs[0], obs[idx], "x", color="green", **marker_params)
        ax.plot(goal[0], goal[idx], "x", color="red", **marker_params)
