import argparse
import time

from isaaclab.app import AppLauncher

# NOTE: We need to run the app launcher first to avoid import errors

# Parse arguments for AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from flow_planning.envs.particle import ParticleEnv

device = "cuda"
save_path = "data/flow_planning/particle_env/dataset.pt"
env = ParticleEnv(
    num_envs=10000,
    grid_size=1.0,
    process_noise=0.02,
    measurement_noise=0.01,
    init_pos_var=0.05,
    kp=2.0,
    kd=1.0,
    dt=0.05,
    seed=42,
    device=device,
)

start_time = time.time()
dataset = env.generate_dataset(
    num_samples=10000,
    trajectory_length=32,
    save_path=save_path,
)
print(
    f"Generated dataset with obs shape {dataset['obs'].shape} samples in {time.time() - start_time:.2f} seconds"
)
print(f"Saved dataset to {save_path}")
# env.visualize_trajectories(dataset, batch_size=100)
