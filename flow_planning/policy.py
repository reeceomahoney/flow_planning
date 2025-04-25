import random

import matplotlib.pyplot as plt
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import isaaclab.utils.math as math_utils
from flow_planning.envs import ParticleEnv
from flow_planning.models.classifier import ClassifierMLP
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
        env: RslRlVecEnvWrapper | ParticleEnv,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_action: int,
        sampling_steps: int,
        lr: float,
        num_iters: int,
        device: str,
    ):
        super().__init__()
        self.env = env
        self.normalizer = normalizer
        self.device = device
        self.isaac_env = isinstance(self.env, RslRlVecEnvWrapper)

        # dims
        self.input_dim = act_dim + obs_dim
        self.action_dim = act_dim
        self.T = T
        self.T_action = T_action

        # diffusion / flow matching
        self.sampling_steps = sampling_steps
        self.guide_scale = 0.0
        self.use_refinement = False

        self.urdf_chain = pk.build_serial_chain_from_urdf(
            open("data/franka_panda/panda.urdf", mode="rb").read(), "panda_hand"
        ).to(device=env.device)

        self._create_model(model, lr, num_iters)
        self.to(device)

    def _create_model(self, model, lr, num_iters):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

    ############
    # Main API #
    ############

    @torch.no_grad()
    def act(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        data = self.process(data)
        x = self.forward(data)
        return {"action": x[..., :7], "traj": x}

    def update(self, data):
        data = self.process(data)
        x_1 = data["traj"]
        x_0 = torch.randn_like(x_1)

        if self.use_refinement and random.random() < 0.5:
            x_0 = torch.cat([x_0[:, : self.T // 2], x_0[:, self.T // 2 :]], dim=0)
            x_1 = torch.cat([x_1[:, : self.T // 2], x_1[:, self.T // 2 :]], dim=0)
            data["obs"] = x_1[:, 0, self.action_dim :]
            data["goal"] = x_1[:, -1, self.action_dim + 18 :]

        bsz = x_1.shape[0]

        # compute sample and target
        t = torch.rand(bsz, 1, 1).to(self.device)
        x_t = (1 - t) * x_0 + t * x_1
        target = x_1 - x_0

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
    def test(self, data: dict) -> float:
        data = self.process(data)
        x = self.forward(data)
        # calculate loss
        traj = self.normalizer.inverse_scale_obs(data["traj"])
        return F.mse_loss(x, traj).item()

    #####################
    # Inference backend #
    #####################

    def forward(self, data: dict[str, Tensor]) -> torch.Tensor:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
        timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)

        # inference
        for i in range(self.sampling_steps):
            x = self.inpaint(x, data)

            t_start = expand_t(timesteps[i], bsz)
            t_end = expand_t(timesteps[i + 1], bsz)
            x += (t_end - t_start) * self.model(x, t_start, data)

            # guidance
            if self.guide_scale > 0:
                grad = self._guide_fn(x, timesteps[i + 1], data)
                dt = timesteps[i + 1] - timesteps[i]
                weight = self.guide_scale * (1 - timesteps[i + 1])
                x += weight * dt * grad

        x = self.inpaint(x, data)

        # refinement step
        if self.use_refinement:
            midpoint = x[:, self.T // 2, self.action_dim :].clone()
            x_0 = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
            x = 0.5 * x_0 + 0.5 * x
            x = torch.cat([x[:, : self.T // 2], x[:, self.T // 2 :]], dim=0)

            data = {k: torch.cat([v] * 2) for k, v in data.items()}
            data["obs"][bsz:] = midpoint
            data["goal"][:bsz] = midpoint[:, 18:27]

            for i in range(self.sampling_steps // 2):
                x = self.inpaint(x, data)

                t_start = expand_t(timesteps[i + self.sampling_steps // 2], 2 * bsz)
                t_end = expand_t(timesteps[i + 1 + self.sampling_steps // 2], 2 * bsz)
                x += (t_end - t_start) * self.model(x, t_start, data)

            x = self.inpaint(x, data)
            x = torch.cat([x[:bsz], x[bsz:]], dim=1)

        # denormalize
        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_obs(x)

    @torch.enable_grad()
    def _guide_fn(self, x: Tensor, t: Tensor, data: dict[str, Tensor]) -> Tensor:
        # collision
        x = x.detach().clone().requires_grad_(True)
        pts = torch.tensor([0.5, 0, 0.2]).to(self.device).expand(x.shape[0], 3)
        th = self.urdf_chain.forward_kinematics(x[0, :, :7], end_only=False)
        matrices = {k: v.get_matrix() for k, v in th.items()}
        pos = {k: v[:, :3, 3] for k, v in matrices.items()}
        pos = torch.stack(list(pos.values()), dim=1)
        dists = torch.norm(pos - pts, dim=-1)
        collision_grad = torch.autograd.grad([dists.sum()], [x], create_graph=True)[0]
        collision_grad.detach_()

        # smoothness
        x = x.detach().clone().requires_grad_(True)
        cost = self.gp(x)
        smooth_grad = torch.autograd.grad([cost.sum()], [x], create_graph=True)[0]
        smooth_grad.detach_()

        return collision_grad - 1e-5 * smooth_grad

    ###################
    # Data processing #
    ###################

    def process(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        data = self.dict_to_device(data)

        if "action" in data:
            # train and test case
            obs = data["obs"][:, 0]
            traj = self.normalizer.scale_obs(data["obs"])
            returns = calculate_return(data["obs"])
            returns = self.normalizer.scale_return(returns)
        else:
            # sim case
            traj = None
            obs = data["obs"]
            returns = torch.ones(obs.shape[0], 1).to(self.device)

        obs = self.normalizer.scale_obs(obs)
        goal = self.normalizer.scale_goal(data["goal"])

        out = {"obs": obs, "goal": goal, "returns": returns}
        if traj is not None:
            out["traj"] = traj
        return out

    def dict_to_device(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device) for k, v in data.items()}

    def inpaint(self, x: Tensor, data: dict[str, Tensor]) -> Tensor:
        x[:, 0] = data["obs"]
        x[:, -1] = data["goal"]
        return x

    #################
    # Visualization #
    #################

    def plot(self, it: int = 0, log: bool = True):
        # get obs and goal
        obs, _ = self.env.get_observations()
        goal = get_goal(self.env)
        # create figure
        guide_scales = torch.tensor([0, 1, 2, 3, 4])
        # projection = "3d" if self.isaac_env else None
        projection = None
        plt.rcParams.update({"font.size": 24})
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(projection=projection)
        if len(guide_scales) > 1:
            colors = plt.get_cmap("viridis")(torch.linspace(0, 1, len(guide_scales)))
        else:
            colors = ["red"]

        # plot trajectories
        for i in range(len(guide_scales)):
            self.guide_scale = guide_scales[i].item()
            traj = self.act({"obs": obs, "goal": goal})["traj"]

            # fk to get hand pose
            th = self.urdf_chain.forward_kinematics(traj[0, :, :7])
            m = th.get_matrix()
            pos = m[:, :3, 3]
            rot = pk.matrix_to_quaternion(m[:, :3, :3])

            # get end effector position
            pos_offset = torch.tensor([[0, 0, 0.107]]).to(self.device)
            rot_offset = torch.tensor([[1, 0, 0, 0]]).to(self.device)
            ee_pos, _ = math_utils.combine_frame_transforms(
                pos, rot, pos_offset, rot_offset
            )

            ee_goal = torch.tensor([0.5, 0.3, 0.2])
            label = f"Scale: {guide_scales[i]}" if len(guide_scales) > 1 else None
            self._draw_trajectory(
                ax, ee_pos, ee_pos[0], ee_goal, color=colors[i], label=label
            )
        self.guide_scale = 0

        # format plot
        if len(guide_scales) > 1:
            ax.legend(loc="upper left", fontsize=20)
        # ax.axis("equal")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        if projection == "3d":
            ax.set_zticklabels([])  # type: ignore
        # ax.set_xlim(0.375, 0.725)
        # ax.set_ylim(0.16, 0.66)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # save or log
        if log:
            name = "Guided Trajectory" if len(guide_scales) > 1 else "Trajectory"
            wandb.log({name: wandb.Image(fig)}, step=it)
            plt.close()
        else:
            plt.savefig("data.pdf", bbox_inches="tight")
            plt.show()

    def _draw_trajectory(self, ax, traj, obs, goal, color=None, label=None):
        traj, obs, goal = traj.cpu(), obs.cpu(), goal.cpu()
        marker_params = {
            "markersize": 35,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 2,
            "linestyle": "None",
        }

        if self.isaac_env:
            c = torch.linspace(0, 1, len(traj)) ** 0.5
            s = [500] * len(traj)
            ax.scatter(traj[:, 1], traj[:, 2], s=s, color=color, label=label)
            ax.plot(obs[1], obs[2], "o", **marker_params)
            ax.plot(goal[1], goal[2], "o", **marker_params)
            marker_params["markersize"] = 20
            marker_params["markerfacecolor"] = "black"
            ax.plot(obs[1], obs[2], "o", **marker_params)
            marker_params["markersize"] = 25
            ax.plot(goal[1], goal[2], "*", **marker_params)
        else:
            c = torch.linspace(0, 1, len(traj)) ** 0.7
            ax.scatter(traj[:, 0], traj[:, 1], c=c, cmap="Reds", s=500)
            ax.plot(obs[0], obs[1], "o", **marker_params)
            ax.plot(goal[0], goal[1], "*", **marker_params)


class ClassifierPolicy(Policy):
    def _create_model(self, model, lr, num_iters):
        self.model = model
        self.classifier = ClassifierMLP(self.input_dim, self.device)
        self.optimizer = AdamW(self.classifier.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

    def update(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised samples
        samples = self.batched_forward(data)
        x = samples["x"].reshape(-1, self.T, self.input_dim)
        t = samples["t"].reshape(-1, 1).expand(-1, data["obs"].shape[0])
        t = t.reshape(-1)
        # compute model output
        pred_value = self.classifier(x, t)

        # calculate loss
        ret = data["returns"].unsqueeze(0)
        ret = ret.expand(self.sampling_steps + 1, -1, -1, -1)
        ret = ret.reshape(-1, self.T, 1)
        loss = F.mse_loss(pred_value, ret)

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def test(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised samples
        samples = self.batched_forward(data)
        x = samples["x"].reshape(-1, self.T, self.input_dim)
        t = samples["t"].reshape(-1, 1).expand(-1, data["obs"].shape[0])
        t = t.reshape(-1)
        # compute model output
        pred_value = self.classifier(x, t)

        # calculate loss
        ret = data["returns"].unsqueeze(0)
        ret = ret.expand(self.sampling_steps + 1, -1, -1, -1)
        ret = ret.reshape(-1, self.T, 1)
        return F.mse_loss(pred_value, ret).item()

    def _guide_fn(self, x: Tensor, t: Tensor, data: dict[str, Tensor]) -> Tensor:
        x = x.detach().clone().requires_grad_(True)
        y = self.classifier(x, t)
        grad = torch.autograd.grad([y.sum()], [x], create_graph=True)[0]
        assert grad is not None
        return grad[..., : self.input_dim].detach()

    @torch.no_grad()
    def batched_forward(self, data: dict) -> dict:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
        time_steps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)

        samples = {"x": [], "t": time_steps}

        # inference
        for i in range(self.sampling_steps):
            x = self.inpaint(x, data)
            samples["x"].append(x)

            x = self.step(x, time_steps[i], time_steps[i + 1], data)

        x = self.inpaint(x, data)
        samples["x"].append(x)
        samples["x"] = torch.stack(samples["x"], dim=0)

        return samples
