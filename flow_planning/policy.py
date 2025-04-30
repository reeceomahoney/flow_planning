import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import isaaclab.utils.math as math_utils
from flow_planning.envs import ParticleEnv
from flow_planning.models.classifier import ClassifierMLP
from flow_planning.models.unet import ConditionalUnet1D
from flow_planning.models.vae import VAE
from flow_planning.utils import CostGPTrajectory, Normalizer, calculate_return, get_goal
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


def expand_t(tensor: Tensor, bsz: int) -> Tensor:
    return tensor.view(1, -1, 1).expand(bsz, -1, 1)


class Policy(nn.Module):
    def __init__(
        self,
        model: ConditionalUnet1D | VAE,
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
        algo: str,
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
        self.train_splitting = True
        self.test_splitting = False
        self.scheduler = DDPMScheduler(self.sampling_steps)
        self.algo = algo

        self.urdf_chain = pk.build_serial_chain_from_urdf(
            open("data/franka_panda/panda.urdf", mode="rb").read(), "panda_hand"
        ).to(device=env.device)
        self.gp = CostGPTrajectory(self.T, 1 / 30, 1)

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

    def reset(self):
        pass

    def update(self, data):
        data = self.process(data)
        x_1 = data["traj"]
        x_0 = torch.randn_like(x_1)

        if self.train_splitting and random.random() < 0.5:
            x_0 = torch.cat([x_0[:, : self.T // 2], x_0[:, self.T // 2 :]], dim=0)
            x_1 = torch.cat([x_1[:, : self.T // 2], x_1[:, self.T // 2 :]], dim=0)
            data["obs"] = x_1[:, 0]
            data["goal"] = x_1[:, -1]

        x_t, target, t = self._compute_sample(x_1, x_0)

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

    def _compute_sample(self, x_1: Tensor, x_0: Tensor) -> tuple:
        bsz = x_1.shape[0]
        if self.algo == "flow_planning":
            t = torch.rand(bsz, 1, 1).to(self.device)
            x_t = (1 - t) * x_0 + t * x_1
            target = x_1 - x_0
        elif self.algo == "mpd":
            t = torch.randint(0, self.sampling_steps, (bsz, 1, 1)).to(self.device)
            x_t = self.scheduler.add_noise(x_1, x_0, t)  # type: ignore
            target = x_0
        else:
            raise NotImplementedError(f"Unknown algorithm: {self.algo}")
        return x_t, target, t

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
        bsz = data["obs"].shape[0]
        # sample noise
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
        self._compute_timesteps()

        # inference
        for i in range(self.sampling_steps):
            x = self.inpaint(x, data)
            x = self._step(x, i, data)

        x = self.inpaint(x, data)

        # refinement step
        if self.test_splitting:
            midpoint = x[:, self.T // 2].clone()
            x_0 = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
            x = 0.5 * x_0 + 0.5 * x
            x = torch.cat([x[:, : self.T // 2], x[:, self.T // 2 :]], dim=0)

            data = {k: torch.cat([v] * 2) for k, v in data.items()}
            data["obs"][bsz:] = midpoint
            data["goal"][:bsz] = midpoint

            for i in range(self.sampling_steps // 2):
                x = self.inpaint(x, data)
                x = self._step(x, i + self.sampling_steps // 2, data)

            x = self.inpaint(x, data)
            x = torch.cat([x[:bsz], x[bsz:]], dim=1)

        # denormalize
        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_obs(x)

    def _compute_timesteps(self):
        if self.algo == "flow_planning":
            self.timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(
                self.device
            )
        elif self.algo == "mpd":
            self.scheduler.set_timesteps(self.sampling_steps, self.device)
            self.timesteps = self.scheduler.timesteps
        else:
            raise NotImplementedError(f"Unknown algorithm: {self.algo}")

    def _step(self, x: Tensor, idx: int, data: dict[str, Tensor]) -> Tensor:
        bsz = data["obs"].shape[0]
        if self.algo == "flow_planning":
            t = expand_t(self.timesteps[idx], bsz)
            t_end = expand_t(self.timesteps[idx + 1], bsz)
            x += (t_end - t) * self.model(x, t, data)
            var = 1 - t_end
        elif self.algo == "mpd":
            t = self.timesteps[idx]
            t_ = t.view(-1, 1, 1).expand(bsz, 1, 1).float()
            out = self.model(x, t_, data)
            x = self.scheduler.step(out, t, x).prev_sample  # type: ignore
            var = self.scheduler._get_variance(t)
        else:
            raise NotImplementedError(f"Unknown algorithm: {self.algo}")

        # guidance
        if self.guide_scale > 0:
            grad = self._guide_fn(x)
            x += self.guide_scale * var * grad  # type: ignore

        return x

    @torch.enable_grad()
    def _guide_fn(self, x: Tensor) -> Tensor:
        # collision
        x = x.detach().clone().requires_grad_(True)
        # x = self.normalizer.inverse_scale_obs(x)
        pts = torch.tensor([0.5, 0, 0.2]).view(1, 1, -1).to(self.device)
        x_ = x.reshape(-1, self.input_dim)
        th = self.urdf_chain.forward_kinematics(x_[:, :7], end_only=False)
        matrices = {k: v.get_matrix() for k, v in th.items()}  # type: ignore
        pos = {k: v[:, :3, 3] for k, v in matrices.items()}
        pos = torch.stack(list(pos.values()), dim=1)
        pos = pos.reshape(-1, self.T, 3)
        dists = torch.norm(pos - pts, dim=-1)
        collision_grad = torch.autograd.grad([dists.sum()], [x])[0].detach()

        # smoothness
        x = x.detach().clone().requires_grad_(True)
        # x = self.normalizer.inverse_scale_obs(x)
        cost = self.gp(x)
        smooth_grad = torch.autograd.grad([cost.sum()], [x])[0].detach()

        return 0.1 * collision_grad - 1e-6 / self.guide_scale * smooth_grad

    ###################
    # Data processing #
    ###################

    def process(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        data = self.dict_to_device(data)

        if "action" in data:
            # train and test case
            obs = data["obs"][:, 0]
            traj = self.normalizer.scale_obs(data["obs"])
            # returns = calculate_return(data["obs"])
            # returns = self.normalizer.scale_return(returns)
            returns = torch.ones(obs.shape[0], 1).to(self.device)
        else:
            # sim case
            traj = None
            obs = data["obs"]
            returns = torch.ones(obs.shape[0], 1).to(self.device)

        obs = self.normalizer.scale_obs(obs)
        goal = self.normalizer.scale_obs(data["goal"])

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

    def compute_ee_pos(self, x: Tensor) -> Tensor:
        # fk to get hand pose
        th = self.urdf_chain.forward_kinematics(x[0, :, :7])
        m = th.get_matrix()  # type: ignore
        pos = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])

        # get end effector position
        pos_offset = torch.tensor([[0, 0, 0.107]]).expand(pos.shape[0], 3)
        rot_offset = torch.tensor([[1, 0, 0, 0]]).expand(rot.shape[0], 4)
        ee_pos, _ = math_utils.combine_frame_transforms(
            pos, rot, pos_offset.to(self.device), rot_offset.to(self.device)
        )
        return ee_pos

    #################
    # Visualization #
    #################

    @torch.no_grad()
    def calculate_goal_error(self) -> tuple[float, float]:
        # fmt: off
        init_pos = torch.tensor([
            # (0.5, -0.3, 0.2)
            [-1.6615e-01, 2.7841e-01, -3.8028e-01, -2.0778e00, 1.3647e-01, 2.3238e00, 1.4746e-01],
            # (0.5, -0.3, 0.6)
            [-2.7545e-01, 2.4703e-01, -3.3734e-01, -1.0436e00, 8.1770e-02, 1.2697e00, 1.9731e-01],
        ]).to(self.device)
        goal_pos = torch.tensor([
            # (0.5, 0.3, 0.2)
            [0.1118,  0.2624,  0.4228, -2.0830, -0.1448,  2.3247,  1.4210],
            # (0.5, 0.3, 0.6)
            [2.5962, -0.5873, -1.4645, -1.1460, -0.6212,  1.2096,  1.6562],
        ]).to(self.device)
        # fmt: on

        # sample data
        indices = torch.randint(0, 2, (64,))
        init_pos_batch = init_pos[indices]
        goal_pos_batch = goal_pos[indices]
        obs = torch.cat([init_pos_batch, torch.zeros_like(init_pos_batch)], dim=-1)
        goal = torch.cat([goal_pos_batch, torch.zeros_like(goal_pos_batch)], dim=-1)

        # compute trajectories
        data = self.process({"obs": obs, "goal": goal})
        traj = self.forward(data)

        # calculate error
        init_error = torch.norm(traj[:, 1, :7] - obs[:, :7], dim=1).mean().item()
        final_error = torch.norm(traj[:, -2, :7] - goal[:, :7], dim=1).mean().item()
        # calculate error std
        init_std = torch.norm(traj[:, 1, :7] - obs[:, :7], dim=1).std().item()
        final_std = torch.norm(traj[:, -2, :7] - goal[:, :7], dim=1).std().item()
        tot_std = (init_std + final_std) / 2
        tot_error = (init_error + final_error) / 2
        return tot_error, tot_std

    def plot(self, it: int = 0, log: bool = True):
        # get obs and goal
        obs = self.env.get_observations()[0][0:1]
        goal = get_goal(self.env)[0:1]
        # create figure
        guide_scales = torch.tensor([0])
        # projection = "3d" if self.isaac_env else None
        projection = None
        plt.rcParams.update({"font.size": 36})
        plt.rcParams.update({"xtick.labelsize": 36, "ytick.labelsize": 36})
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection=projection)
        if len(guide_scales) > 1:
            colors = plt.get_cmap("viridis")(torch.linspace(0, 1, len(guide_scales)))
        else:
            colors = ["red"]

        # plot trajectories
        for i in range(len(guide_scales)):
            self.guide_scale = guide_scales[i].item()
            traj = self.act({"obs": obs, "goal": goal})["traj"]
            # ee_pos = self.compute_ee_pos(traj)
            # ee_goal = torch.tensor([0.5, 0.3, 0.2])
            label = f"Scale: {guide_scales[i]:.1f}" if len(guide_scales) > 1 else None
            # self._draw_trajectory(
            #     ax, ee_pos, ee_pos[0], ee_goal, color=colors[i], label=label
            # )
            self._draw_trajectory(
                ax, traj, obs, goal, color=colors[i], label=label
            )
        self.guide_scale = 0

        # format plot
        # if len(guide_scales) > 1:
        #     ax.legend(loc="upper left", fontsize=20)
        # ax.axis("equal")
        # ax.set_xlabel("Y")
        # ax.set_ylabel("Z")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        if projection == "3d":
            ax.set_zticklabels([])  # type: ignore
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(True, alpha=0.6)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
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
            "markersize": 55,
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
            traj, obs, goal = traj[0], obs[0], goal[0]
            c = torch.linspace(0, 1, len(traj)) ** 0.7
            ax.scatter(traj[:, 0], traj[:, 1], c=c, cmap="Blues", s=1500)
            ax.plot(obs[0], obs[1], "o", **marker_params)
            ax.plot(goal[0], goal[1], "o", **marker_params)
            marker_params["markersize"] = 35
            marker_params["markerfacecolor"] = "black"
            ax.plot(obs[0], obs[1], "o", **marker_params)
            marker_params["markersize"] = 45
            ax.plot(goal[0], goal[1], "*", **marker_params)

            # position = traj[:, :2].detach().cpu().numpy()
            # points = position.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #
            # t = np.linspace(1, 0, len(segments))
            # lc = LineCollection(
            #     segments,
            #     cmap="RdBu",
            #     norm=Normalize(0, 1),
            #     array=t,
            #     linewidths=20,
            #     capstyle="round",
            # )
            # ax.add_collection(lc)


# We need this becuase the guidance code isn't scriptable
class JitPolicy(Policy):
    def _create_model(self, model, lr, num_iters):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)

    def forward(self, x: Tensor, data: dict[str, Tensor], idx: int) -> torch.Tensor:
        # process
        data = self.process(data)
        x = self.inpaint(x, data)

        # single inference step
        bsz = data["obs"].shape[0]
        t = expand_t(self.timesteps[idx], bsz)
        t_end = expand_t(self.timesteps[idx + 1], bsz)
        x += (t_end - t) * self.model(x, t, data)

        return self.inpaint(x, data)


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
