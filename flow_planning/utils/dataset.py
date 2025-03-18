import logging
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

log = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    def __init__(
        self,
        data_directory: str | None,
        T: int,
        T_cond: int,
        task_name: str | None,
        device="cpu",
    ):
        self.T = T
        self.T_cond = T_cond
        self.device = device

        if data_directory is not None:
            # build path
            current_dir = os.path.dirname(os.path.realpath(__file__))
            dataset_path = current_dir + "/../../" + data_directory
            log.info(f"Loading data from {data_directory}")

            # load data
            data = {}
            with h5py.File(dataset_path, "r") as f:

                def load_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[:]

                f.visititems(load_dataset)

            # (B, T, D)
            data = {
                k: torch.from_numpy(v).transpose(0, 1).to(device)
                for k, v in data.items()
            }
            obs = data["observations"]
            # remove commands
            # obs = torch.cat([obs[..., :27], obs[..., 34:]], dim=-1)
            obs = obs[..., 18:21]
            actions = data["actions"]
            terminals = data["terminals"]
            split_indices = torch.where(terminals.flatten() == 1)[0] + 1

            obs_splits = self.split_eps(obs, split_indices)
            actions_splits = self.split_eps(actions, split_indices)

        else:
            dataset_path = f"data/diffusion/maze/dataset_{task_name}.pkl"
            if os.path.exists(dataset_path):
                # load pre-processed dataset
                obs_splits, actions_splits = self.load_dataset(dataset_path)  # type: ignore
                log.info("Loaded pre-processed dataset")
            else:
                # process the dataset
                dataset_name = self.get_dataset_name(task_name)
                dataset = minari.load_dataset(dataset_name)
                obs_splits, actions_splits = [], []
                for episode in dataset:
                    obs_splits.append(
                        torch.tensor(
                            episode.observations["observation"], dtype=torch.float
                        )
                    )
                    actions_splits.append(
                        torch.tensor(episode.actions, dtype=torch.float)
                    )

                # save the dataset to speedup launch
                self.save_dataset(obs_splits, actions_splits, dataset_path)

        self.calculate_norm_data(obs_splits, actions_splits)

        # add padding to make all sequences the same length
        max_len = max(split.shape[0] for split in obs_splits)
        obs = self.add_padding(obs_splits, max_len, temporal=True)
        actions = self.add_padding(actions_splits, max_len, temporal=True)
        masks = self.create_masks(obs_splits, max_len)

        # TODO: this is a hack, get the real last timestep
        goal = obs[:, -1:].expand(-1, obs.shape[1], -1)

        self.data = {"obs": obs, "action": actions, "mask": masks, "goal": goal}

        obs_size = list(self.data["obs"].shape)
        action_size = list(self.data["action"].shape)
        log.info(f"Dataset size | Observations: {obs_size} | Actions: {action_size}")

    def __len__(self):
        return len(self.data["obs"])

    def __getitem__(self, idx):
        return {
            "obs": self.data["obs"][idx],
            "action": self.data["action"][idx],
            "mask": self.data["mask"][idx],
            "goal": self.data["goal"][idx],
        }

    def split_eps(self, x, split_indices):
        x = torch.tensor_split(x.reshape(-1, x.shape[-1]), split_indices.tolist())
        # remove last empty split
        return x[:-1]

    def add_padding(self, splits, max_len, temporal):
        x = []

        # Make all sequences the same length
        for split in splits:
            padded_split = torch.nn.functional.pad(
                split, (0, 0, 0, max_len - split.shape[0])
            )
            x.append(padded_split)
        x = torch.stack(x)

        if temporal:
            # Add initial padding to handle episode starts
            x_pad = torch.zeros_like(x[:, : self.T_cond - 1, :])
            x = torch.cat([x_pad, x], dim=1)
        else:
            # For non-temporal data, e.g. skills, just take the first timestep
            x = x[:, 0]

        return x.to(self.device)

    def create_masks(self, splits, max_len):
        masks = []

        # Create masks to indicate the padding values
        for split in splits:
            mask = torch.concatenate(
                [torch.ones(split.shape[0]), torch.zeros(max_len - split.shape[0])]
            )
            masks.append(mask)
        masks = torch.stack(masks)

        # Add initial padding to handle episode starts
        masks_pad = torch.ones((masks.shape[0], self.T_cond - 1))
        masks = torch.cat([masks_pad, masks], dim=1)

        return masks.to(self.device)

    def calculate_norm_data(self, obs_splits, actions_splits):
        all_obs = torch.cat(obs_splits)
        all_actions = torch.cat(actions_splits)
        # all_obs_acts = torch.cat([all_actions, all_obs], dim=-1)
        all_obs_acts = all_obs

        self.x_mean = all_obs.mean(0)
        self.x_std = all_obs.std(0)
        self.x_min = all_obs.min(0).values
        self.x_max = all_obs.max(0).values

        self.y_mean = all_obs_acts.mean(0)
        self.y_std = all_obs_acts.std(0)
        self.y_min = all_obs_acts.min(0).values
        self.y_max = all_obs_acts.max(0).values

    # Save the dataset
    def save_dataset(self, obs_splits, actions_splits, filename="dataset.pkl"):
        with open(filename, "wb") as f:
            pickle.dump((obs_splits, actions_splits), f)

    # Load the dataset if it exists
    def load_dataset(self, filename="dataset.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return None

    def get_dataset_name(self, task_name):
        difficulty = task_name.split("_")[1].lower().split("-")[0]
        return f"D4RL/pointmaze/{difficulty}-v2"


class SlicerWrapper(Dataset):
    def __init__(self, dataset: Subset, T_cond: int, T: int):
        self.dataset = dataset
        self.T_cond = T_cond
        self.T = T
        self.slices = self._create_slices(T_cond, T)

    def _create_slices(self, T_cond, T):
        slices = []
        window = T_cond + T - 1
        for i in range(len(self.dataset)):
            length = self.dataset[i]["mask"].sum().int().item()
            if length >= window:
                slices += [
                    (i, start, start + window) for start in range(length - window + 1)
                ]
            else:
                # add a padded slice
                slices += [(i, start, start + window) for start in range(length - 1)]
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        x = self.dataset[i]
        return {k: v[start:end] for k, v in x.items()}


class PDControlledParticleDataset(Dataset):
    """
    Dataset for particle trajectories controlled by a PD controller.

    Particles start near either bottom-left or bottom-right corner with
    variation in initial position, and are guided to the opposite corner
    via PD control with noise in the dynamics.
    """

    def __init__(
        self,
        num_samples=1000,
        trajectory_length=100,
        grid_size=1.0,
        process_noise=0.02,
        measurement_noise=0.01,
        init_pos_var=0.05,
        kp=1.0,
        kd=0.5,
        dt=0.1,
        seed=42,
    ):
        """
        Args:
            num_samples: Number of trajectory samples in the dataset
            trajectory_length: Number of steps in each trajectory
            grid_size: Size of the grid (normalized to 1.0)
            process_noise: Standard deviation of noise in the dynamics
            measurement_noise: Standard deviation of noise in the measurement
            init_pos_var: Standard deviation of the initial position variation
            kp: Proportional gain for the PD controller
            kd: Derivative gain for the PD controller
            dt: Time step for the simulation
            seed: Random seed for reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.grid_size = grid_size
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.init_pos_var = init_pos_var
        self.kp = kp
        self.kd = kd
        self.dt = dt

        # Generate the dataset
        self.trajectories = []
        self.start_corners = []
        self.velocities = []  # Store velocities for analysis
        self.accelerations = []  # Store accelerations for analysis
        self.control_signals = []  # Store control signals for analysis

        if os.path.exists("data/diffusion/pd_env/dataset.pt"):
            data = torch.load("data/diffusion/pd_env/dataset.pt")
            self.obs = data["obs"]
            self.actions = data["actions"]
        else:
            for i in range(num_samples):
                # Randomly choose starting corner (0: bottom-left, 1: bottom-right)
                start_corner = i % 2

                # Generate the trajectory with PD control
                trajectory, velocity, acceleration, control = (
                    self._generate_pd_trajectory(start_corner)
                )

                self.trajectories.append(trajectory)
                self.start_corners.append(start_corner)
                self.velocities.append(velocity)
                self.accelerations.append(acceleration)
                self.control_signals.append(control)

            # Convert lists to tensors
            # self.obs = torch.stack(self.trajectories)
            self.obs = torch.cat(
                [torch.stack(self.trajectories), torch.stack(self.velocities)], dim=-1
            )
            self.start_corners = torch.tensor(self.start_corners)
            self.accelerations = torch.stack(self.accelerations)
            self.actions = torch.stack(self.control_signals)

            data = {"obs": self.obs, "actions": self.actions}
            torch.save(data, "data/diffusion/pd_env/dataset.pt")

        self.calculate_norm_data(self.obs, self.actions)

    def _generate_pd_trajectory(self, start_corner):
        """Generate a single trajectory with PD control"""
        if start_corner == 0:
            # Bottom-left to top-right
            # Add variation to the initial position
            init_pos = torch.tensor([0.0, 0.0]) + torch.randn(2) * self.init_pos_var
            init_pos = torch.clamp(
                init_pos, 0, self.grid_size * 0.2
            )  # Keep it in the corner region
            target_pos = torch.tensor([self.grid_size, self.grid_size])
        else:
            # Bottom-right to top-left
            # Add variation to the initial position
            init_pos = (
                torch.tensor([0.0, self.grid_size]) + torch.randn(2) * self.init_pos_var
            )
            init_pos = torch.clamp(
                init_pos,
                torch.tensor([0.0, self.grid_size * 0.8]),
                torch.tensor([self.grid_size * 0.2, self.grid_size]),
            )
            target_pos = torch.tensor([self.grid_size, 0.0])

        # Initialize trajectory arrays
        trajectory = torch.zeros((self.trajectory_length, 2))
        velocity = torch.zeros((self.trajectory_length, 2))
        acceleration = torch.zeros((self.trajectory_length, 2))
        control_signal = torch.zeros((self.trajectory_length, 2))

        # Set initial state
        trajectory[0] = init_pos
        velocity[0] = torch.zeros(2)  # Start with zero velocity

        # Simulation loop with PD control
        for t in range(1, self.trajectory_length):
            # Current state
            current_pos = trajectory[t - 1]
            current_vel = velocity[t - 1]

            # Error and error derivative
            error = target_pos - current_pos
            error_derivative = -current_vel  # Assuming target velocity is zero

            # PD control law
            control = self.kp * error + self.kd * error_derivative
            control_signal[t] = control

            # Calculate acceleration (F = ma, assuming m = 1)
            acc = control + torch.randn(2) * self.process_noise
            acceleration[t] = acc

            # Update velocity: v = v0 + a * dt
            new_vel = current_vel + acc * self.dt
            velocity[t] = new_vel

            # Update position: p = p0 + v * dt
            new_pos = current_pos + new_vel * self.dt

            # Add measurement noise
            noisy_pos = new_pos + torch.randn(2) * self.measurement_noise

            # Clip position to grid boundaries
            trajectory[t] = torch.clamp(noisy_pos, 0, self.grid_size)

        return trajectory, velocity, acceleration, control_signal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            trajectory: Tensor of shape [trajectory_length, 2] containing (x,y) coordinates
            start_corner: 0 for bottom-left, 1 for bottom-right
            velocity: Tensor of shape [trajectory_length, 2] containing velocity vectors
            acceleration: Tensor of shape [trajectory_length, 2] containing acceleration vectors
            control_signal: Tensor of shape [trajectory_length, 2] containing control signals
        """
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "goal": self.obs[idx, -1],
            "mask": torch.ones(self.trajectory_length),
        }

    def calculate_norm_data(self, all_obs, all_actions):
        all_obs = all_obs.reshape(-1, all_obs.shape[-1])
        all_actions = all_actions.reshape(-1, all_actions.shape[-1])
        all_obs_acts = torch.cat([all_actions, all_obs], dim=-1)

        self.x_mean = all_obs.mean(0)
        self.x_std = all_obs.std(0)
        self.x_min = all_obs.min(0).values
        self.x_max = all_obs.max(0).values

        self.y_mean = all_obs_acts.mean(0)
        self.y_std = all_obs_acts.std(0)
        self.y_min = all_obs_acts.min(0).values
        self.y_max = all_obs_acts.max(0).values

    def visualize_batch(self, batch_size=10):
        """Visualize a batch of trajectories"""
        plt.figure(figsize=(12, 10))

        for i in range(min(batch_size, self.num_samples)):
            sample = self[i]
            trajectory = sample["obs"][:, :2]

            plt.plot(trajectory[:, 0], trajectory[:, 1], c="blue", alpha=0.6)
            plt.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=30)
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=30)

        plt.title(f"Batch of {batch_size} PD Controlled Particle Trajectories")
        plt.xlim(-0.1, self.grid_size + 0.1)
        plt.ylim(-0.1, self.grid_size + 0.1)
        plt.grid(True)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Bottom-Left Start"),
            Line2D([0], [0], color="orange", lw=2, label="Bottom-Right Start"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=8,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="End",
            ),
        ]
        plt.legend(handles=legend_elements)

        plt.show()


def get_dataloaders(
    task_name: str,
    data_directory: str,
    T_cond: int,
    T: int,
    train_fraction: float,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    test: bool = False,
):
    # Build the datasets
    # dataset = PDControlledParticleDataset(
    #     num_samples=100 if test else 10000,
    #     trajectory_length=32,
    #     grid_size=1.0,
    #     process_noise=0.03,
    #     measurement_noise=0.01,
    #     init_pos_var=0.05,
    #     kp=2.0,
    #     kd=1.0,
    #     dt=0.05,
    #     seed=42,
    # )
    dataset = ExpertDataset(
        data_directory=data_directory,
        T=T,
        T_cond=T_cond,
        task_name=task_name,
    )
    train, val = random_split(dataset, [train_fraction, 1 - train_fraction])
    train_set = SlicerWrapper(train, T_cond, T)
    test_set = SlicerWrapper(val, T_cond, T)

    # Build the dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # calculate value range
    gammas = torch.tensor([0.99**i for i in range(T)])
    returns = []
    # for batch in train_dataloader:
    #     obs = batch["obs"]
    #     mask = batch["mask"]
    #     goal = batch["goal"]
    #     # goal = sample_goal_poses_from_list(obs.shape[0], obs.device)
    #     returns.append(
    #         calculate_return(obs[..., 18:21], obs[:, 0, 18:21], goal, mask, gammas)
    #     )
    # returns = torch.cat(returns)
    returns = torch.ones(1)

    dl = train_dataloader.dataset.dataset.dataset
    dl.r_max = returns.max()
    dl.r_min = returns.min()

    return train_dataloader, test_dataloader
