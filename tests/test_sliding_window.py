import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from locodiff.utils import bidirectional_sliding_window_scheduler


def test_bidirectional_scheduler():
    """
    Test the bidirectional sliding window scheduler with a fully vectorized PyTorch implementation.
    """
    # Parameters
    trajectory_length = 60
    num_timesteps = 50

    # Generate global timesteps
    global_timesteps = torch.linspace(0, 1, num_timesteps).unsqueeze(1)

    # Generate data for heatmap in a single call (much more efficient)
    data = bidirectional_sliding_window_scheduler(global_timesteps, trajectory_length)

    # Convert to numpy for matplotlib
    data_np = data.numpy()

    # Create custom colormap: blue (0.0, noisy) to white (1.0, clean)
    colors = [(0, 0, 1), (0.5, 0.5, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("NoiseToClean", colors)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot heatmap
    heatmap = plt.imshow(
        data_np,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, trajectory_length - 1, 0, 1],
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Local Timestep Value (0=noisy, 1=clean)")

    # Add labels and title
    plt.xlabel("Trajectory Position")
    plt.ylabel("Global Timestep")
    plt.title("Bidirectional Sliding Window Scheduler (PyTorch Implementation)")

    # Add gridlines
    plt.grid(False)

    # Add text annotations to explain the visualization
    plt.figtext(
        0.02,
        0.02,
        "Visualization shows how denoising progresses from the edges toward the center.\n"
        "Blue = noisy (timestep=0), White = clean (timestep=1)\n"
        "Each position transitions from noisy to clean in exactly 10 steps.",
        fontsize=9,
    )

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_bidirectional_scheduler()
