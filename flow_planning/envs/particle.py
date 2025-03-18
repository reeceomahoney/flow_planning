import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class ParticleEnv:
    def __init__(
        self,
        num_envs=64,
        grid_size=1.0,
        process_noise=0.02,
        measurement_noise=0.01,
        init_pos_var=0.05,
        kp=1.0,
        kd=0.5,
        dt=0.1,
        seed=42,
        device="cpu",
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = 4
        self.act_dim = 2
        self.state = torch.zeros((num_envs, 4), device=device)
        self.max_episode_length = 32

        self.num_envs = num_envs
        self.grid_size = grid_size
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.init_pos_var = init_pos_var
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.device = device

    def reset(self, start_corners=None):
        if start_corners is None:
            start_corners = torch.randint(0, 2, (self.num_envs,), device=self.device)

        init_pos = torch.zeros((self.num_envs, 2), device=self.device)
        target_pos = torch.zeros((self.num_envs, 2), device=self.device)

        bl_mask = start_corners == 0
        num_bl = int(bl_mask.sum().item())
        if num_bl > 0:
            bl_pos = torch.zeros((num_bl, 2), device=self.device)
            bl_pos += torch.randn((num_bl, 2), device=self.device) * self.init_pos_var
            bl_pos = torch.clamp(bl_pos, 0, self.grid_size * 0.2)
            init_pos[bl_mask] = bl_pos
            target_pos[bl_mask] = self.grid_size

        br_mask = start_corners == 1
        num_br = int(br_mask.sum().item())
        if num_br > 0:
            br_pos = torch.zeros((num_br, 2), device=self.device)
            br_pos[:, 1] = self.grid_size
            br_pos += torch.randn((num_br, 2), device=self.device) * self.init_pos_var
            br_pos[:, 0] = torch.clamp(br_pos[:, 0], 0, self.grid_size * 0.2)
            br_pos[:, 1] = torch.clamp(
                br_pos[:, 1], self.grid_size * 0.8, self.grid_size
            )
            init_pos[br_mask] = br_pos
            target_pos[br_mask, 0] = self.grid_size
            target_pos[br_mask, 1] = 0.0

        init_vel = torch.zeros((self.num_envs, 2), device=self.device)
        state = torch.cat([init_pos, init_vel], dim=1)

        self.state = state
        self.goal = target_pos
        self.start_corners = start_corners
        return self.state

    def get_observations(self):
        return self.state, None

    def compute_pd_control(self):
        pos = self.state[:, :2]
        vel = self.state[:, 2:4]
        error = self.goal - pos
        error_derivative = -vel
        return self.kp * error + self.kd * error_derivative

    def step(self, action):
        pos = self.state[:, :2]
        vel = self.state[:, 2:4]

        action += torch.randn(self.num_envs, 2, device=self.device) * self.process_noise
        new_vel = vel + action * self.dt
        new_pos = pos + new_vel * self.dt

        noisy_pos = (
            new_pos
            + torch.randn(self.num_envs, 2, device=self.device) * self.measurement_noise
        )
        clipped_pos = torch.clamp(noisy_pos, 0, self.grid_size)
        new_state = torch.cat([clipped_pos, new_vel], dim=1)

        self.state = new_state
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device)

        return self.state, rewards, dones, {}

    def generate_trajectories(self, trajectory_length, start_corners=None):
        obs = torch.zeros((self.num_envs, trajectory_length, self.obs_dim), device=self.device)
        actions = torch.zeros((self.num_envs, trajectory_length, self.act_dim), device=self.device)

        self.reset(start_corners)

        obs[:, 0, :] = self.state

        for t in range(1, trajectory_length):
            control = self.compute_pd_control()
            actions[:, t, :] = control

            next_state = self.step(control)[0]

            obs[:, t, :] = next_state

        return {"obs": obs, "actions": actions}

    def generate_dataset(self, num_samples, trajectory_length, save_path=None):
        remaining = num_samples
        all_obs = []
        all_actions = []

        while remaining > 0:
            current_batch = min(self.num_envs, remaining)
            self.num_envs = current_batch

            start_idx = num_samples - remaining
            start_corners = torch.tensor(
                [(start_idx + i) % 2 for i in range(current_batch)], device=self.device
            )

            results = self.generate_trajectories(trajectory_length, start_corners)

            all_obs.append(results["obs"].cpu())
            all_actions.append(results["actions"].cpu())

            remaining -= current_batch

        obs = torch.cat(all_obs, dim=0)
        actions = torch.cat(all_actions, dim=0)

        result = {"obs": obs, "actions": actions}

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(result, save_path)

        return result

    def load_dataset(self, path, device=None):
        if device is None:
            device = self.device

        data = torch.load(path, map_location=device)
        return {k: v.to(device) for k, v in data.items()}

    def visualize_trajectories(self, trajectories, batch_size=10):
        plt.figure(figsize=(12, 10))

        positions = trajectories["obs"][..., :2]
        n_trajs = min(batch_size, positions.shape[0])

        for i in range(n_trajs):
            position = positions[i].cpu()

            plt.plot(position[:, 0], position[:, 1], c="blue", alpha=0.5)
            plt.scatter(position[0, 0], position[0, 1], c="green", s=30)
            plt.scatter(position[-1, 0], position[-1, 1], c="red", s=30)

        plt.title(f"Batch of {n_trajs} PD Controlled Particle Trajectories")
        plt.xlim(-0.1, self.grid_size + 0.1)
        plt.ylim(-0.1, self.grid_size + 0.1)
        plt.grid(True)
        plt.show()

    def close(self):
        pass
