import logging
import os
import pickle

import h5py
import minari
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .train_utils import calculate_return

log = logging.getLogger(__name__)


class FlowPlanningDataset(Dataset):
    def __init__(self, env_name: str, T: int, device="cpu"):
        self.T = T
        self.device = device

        if env_name == "Isaac-Franka-FlowPlanning":
            # build path
            current_dir = os.path.dirname(os.path.realpath(__file__))
            data_directory = "/data/rsl_rl/stitch_data.hdf5"
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

        elif env_name == "Particle":
            dataset = torch.load("data/flow_planning/particle_env/dataset.pt")
            obs = dataset["obs"]
            actions = dataset["actions"]
            terminals = torch.zeros_like(obs[:, :, 0])
            terminals[:, -1] = 1
            split_indices = torch.where(terminals.flatten() == 1)[0] + 1

            obs_splits = self.split_eps(obs, split_indices)
            actions_splits = self.split_eps(actions, split_indices)

        elif env_name == "Maze":
            dataset_path = f"data/diffusion/maze/dataset_{env_name}.pkl"
            if os.path.exists(dataset_path):
                # load pre-processed dataset
                obs_splits, actions_splits = self.load_dataset(dataset_path)  # type: ignore
                log.info("Loaded pre-processed dataset")
            else:
                # process the dataset
                dataset_name = self.get_dataset_name(env_name)
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

        if not temporal:
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
        return torch.stack(masks).to(self.device)

    def calculate_norm_data(self, obs_splits, actions_splits):
        all_obs = torch.cat(obs_splits)
        all_actions = torch.cat(actions_splits)
        all_obs_acts = torch.cat([all_actions, all_obs], dim=-1)

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
    def __init__(self, dataset: Subset, T: int):
        self.dataset = dataset
        self.slices = self._create_slices(T)

    def _create_slices(self, T):
        slices = []
        window = T
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


def get_dataloaders(
    env_name: str,
    T: int,
    train_fraction: float,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
):
    # Build the datasets
    dataset = FlowPlanningDataset(
        env_name=env_name,
        T=T,
    )
    train, val = random_split(dataset, [train_fraction, 1 - train_fraction])
    train_set = SlicerWrapper(train, T)
    test_set = SlicerWrapper(val, T)

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
    dl = train_dataloader.dataset.dataset.dataset  # type: ignore
    use_value_range = False
    if use_value_range:
        gammas = torch.tensor([0.99**i for i in range(T)])
        returns = []
        for batch in train_dataloader:
            obs = batch["obs"]
            mask = batch["mask"]
            goal = batch["goal"]
            returns.append(
                calculate_return(obs[..., 18:21], obs[:, 0, 18:21], goal, mask, gammas)
            )
        returns = torch.cat(returns)
        dl.r_max = returns.max()
        dl.r_min = returns.min()
    else:
        dl.r_max = torch.tensor(1.0)
        dl.r_min = torch.tensor(0.0)

    return train_dataloader, test_dataloader
