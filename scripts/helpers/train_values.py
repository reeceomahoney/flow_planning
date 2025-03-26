import os

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Define a simple MLP for the value function
class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueFunction, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)


def check_collisions(traj: torch.Tensor) -> torch.Tensor:
    """0 if in collision, 1 otherwise"""
    x_mask = (traj[..., 0] >= 0.55) & (traj[..., 0] <= 0.65)
    y_mask = (traj[..., 1] >= -0.8) & (traj[..., 1] <= 0.8)
    z_mask = (traj[..., 2] >= 0.0) & (traj[..., 2] <= 0.6)
    return ~(x_mask & y_mask & z_mask)


def train_value_function(
    states,
    rewards,
    next_states,
    dones,
    device,
    gamma=0.99,
    lr=1e-4,
    batch_size=64,
    epochs=100,
):
    # Move data to device
    states = states.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    D = states.shape[-1]
    value_function = ValueFunction(D).to(device)
    optimizer = optim.Adam(value_function.parameters(), lr=lr)

    # Reshape data for training: [B*T, D]
    states_flat = states.reshape(-1, D)
    next_states_flat = next_states.reshape(-1, D)
    rewards_flat = rewards.reshape(-1)
    dones_flat = dones.reshape(-1)

    # Create dataset and dataloader
    dataset = TensorDataset(states_flat, rewards_flat, next_states_flat, dones_flat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train = False

    # visualize
    if not train:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        value_function.load_state_dict(torch.load("value_function.pt"))
        batch_states, batch_rewards, batch_next_states, batch_dones = next(
            iter(dataloader)
        )
        values = value_function(batch_states).detach().cpu().numpy()

        # plot value heatmap
        ax.scatter(
            batch_states[:, 7].cpu(),
            batch_states[:, 8].cpu(),
            batch_states[:, 9].cpu(),
            c=values,
            cmap="coolwarm",
        )
        plt.colorbar(ax.scatter([], [], [], c=[], cmap="coolwarm"))
        plt.show()

    # Training loop
    if train:
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_states, batch_rewards, batch_next_states, batch_dones in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
            ):
                current_values = value_function(batch_states)

                # TD target: r + gamma * V(s')
                with torch.no_grad():
                    next_values = value_function(batch_next_states)
                    td_targets = batch_rewards + gamma * (1 - batch_dones) * next_values

                loss = F.mse_loss(td_targets, current_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Print progress
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")
            # save
            torch.save(value_function.state_dict(), "value_function.pt")

        return value_function


# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_directory = "/data/rsl_rl/stitch_data.hdf5"
    dataset_path = current_dir + "/../../" + data_directory

    # load data
    data = {}
    with h5py.File(dataset_path, "r") as f:

        def load_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]

        f.visititems(load_dataset)

    data = {k: torch.from_numpy(v).transpose(0, 1).to(device) for k, v in data.items()}
    obs = data["observations"][..., 18:27]
    actions = data["actions"]
    states = torch.cat([actions, obs], dim=-1)
    dones = data["terminals"]
    rewards = torch.sqrt((obs[..., 0] - 0.6) ** 2 + (obs[..., 1] - 0.2) ** 2)

    # Train value function
    value_function = train_value_function(
        states=states[:, :-1],
        rewards=rewards[:, :-1],
        next_states=states[:, 1:],
        dones=dones[:, :-1],
        device=device,
        batch_size=1024,
        epochs=50,
    )
