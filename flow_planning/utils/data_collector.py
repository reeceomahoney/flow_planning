import os

import h5py
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class DataCollector:
    """A class to collect and store data from gym environments in HDF5 format."""

    def __init__(self, env: RslRlVecEnvWrapper, file_path: str):
        self.env = env
        self.file_path = file_path

        # Initialize buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        self._initialize_h5_file()

    def _initialize_h5_file(self):
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))

        with h5py.File(self.file_path, "w") as f:
            env_group = f.create_group("env_info")
            env_group.attrs["env_id"] = self.env.unwrapped.spec.id  # type: ignore
            env_group.attrs["action_space"] = str(self.env.action_space)
            env_group.attrs["observation_space"] = str(self.env.observation_space)

    def add_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        self.obs_buffer.append(observation.detach().cpu())
        self.action_buffer.append(action.detach().cpu())
        self.reward_buffer.append(reward.detach().cpu())
        self.done_buffer.append(done.detach().cpu())

    def flush(self):
        print("Saving data to disk...")

        if not self.obs_buffer:
            return

        with h5py.File(self.file_path, "a") as f:
            # Stack tensors and convert to numpy
            obs_array = torch.stack(self.obs_buffer).numpy()
            action_array = torch.stack(self.action_buffer).numpy()
            reward_array = torch.stack(self.reward_buffer).numpy()
            done_array = torch.stack(self.done_buffer).numpy()

            # Write data
            f.create_dataset("observations", data=obs_array, compression="gzip")
            f.create_dataset("actions", data=action_array, compression="gzip")
            f.create_dataset("rewards", data=reward_array, compression="gzip")
            f.create_dataset("terminals", data=done_array, compression="gzip")

        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
