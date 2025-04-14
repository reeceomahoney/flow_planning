import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def collision_guide(
    points: torch.Tensor,
    x_min: float,  # Box limits
    x_max: float,
    y_min: float,
    y_max: float,
    max_magnitude: float = 100.0,  # Magnitude for inside/boundary
    decay_rate: float = 0.5,  # Exponential decay rate
    epsilon: float = 1e-12,
) -> torch.Tensor:
    device = points.device
    dtype = points.dtype  # Use dtype from input points tensor

    # Input validation
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid box dimensions: min must be less than max.")

    x = points[:, 0]
    y = points[:, 1]

    F = torch.zeros_like(points)  # Initialize output tensor

    # Determine masks for different regions
    is_strictly_inside = (x_min < x) & (x < x_max) & (y_min < y) & (y < y_max)
    is_outside_or_boundary = ~is_strictly_inside

    # Process Inside Points
    if torch.any(is_strictly_inside):
        idx_inside = torch.where(is_strictly_inside)[0]
        if idx_inside.numel() > 0:
            x_in = x[idx_inside]
            y_in = y[idx_inside]

            # Use scalar box limits directly
            d_left = x_in - x_min
            d_right = x_max - x_in
            d_bottom = y_in - y_min
            d_top = y_max - y_in

            all_d = torch.stack([d_left, d_right, d_bottom, d_top], dim=1)
            _, min_idx = torch.min(all_d, dim=1)  # Find closest wall index

            normals = torch.tensor(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                device=device,
            )
            # Vecor field points outward to the nearest wall
            F[idx_inside] = -max_magnitude * normals[min_idx]

    # Process Outside or Boundary Points
    if torch.any(is_outside_or_boundary):
        idx_out_bound = torch.where(is_outside_or_boundary)[0]
        if idx_out_bound.numel() > 0:
            x_out = x[idx_out_bound]
            y_out = y[idx_out_bound]

            F_out_bound_section = torch.zeros((x_out.shape[0], 2), device=device)

            # Use scalar box limits in clamp
            sx = torch.clamp(x_out, min=x_min, max=x_max)
            sy = torch.clamp(y_out, min=y_min, max=y_max)

            vx = x_out - sx
            vy = y_out - sy
            d_sq = vx * vx + vy * vy

            # Use scalar epsilon for comparison
            is_on_boundary = d_sq < epsilon
            is_valid_outside = ~is_on_boundary

            # --- Calculate for valid outside points (Exponential Decay) ---
            idx_valid_outside_section = torch.where(is_valid_outside)[0]
            if idx_valid_outside_section.numel() > 0:
                vx_valid = vx[idx_valid_outside_section]
                vy_valid = vy[idx_valid_outside_section]
                d_sq_valid = d_sq[idx_valid_outside_section]

                # Use scalar epsilon in clamp
                d_sq_valid_safe = torch.clamp(d_sq_valid, min=epsilon)
                d_valid = torch.sqrt(d_sq_valid_safe)

                # Use scalar parameters for magnitude calculation
                magnitude_exp = max_magnitude * torch.exp(-decay_rate * d_valid)

                ux = vx_valid / d_valid  # Unit direction x
                uy = vy_valid / d_valid  # Unit direction y

                F_out_bound_section[idx_valid_outside_section, 0] = magnitude_exp * ux
                F_out_bound_section[idx_valid_outside_section, 1] = magnitude_exp * uy

            # --- Calculate for boundary points (Apply Fixed Magnitude) ---
            idx_on_boundary_section = torch.where(is_on_boundary)[0]
            if idx_on_boundary_section.numel() > 0:
                x_b = x_out[idx_on_boundary_section]
                y_b = y_out[idx_on_boundary_section]
                num_boundary = x_b.shape[0]
                n_boundary = torch.zeros((num_boundary, 2), device=device, dtype=dtype)

                # Use scalar parameters for boundary checks
                on_left = torch.abs(x_b - x_min) < epsilon
                on_right = torch.abs(x_b - x_max) < epsilon
                on_bottom = torch.abs(y_b - y_min) < epsilon
                on_top = torch.abs(y_b - y_max) < epsilon

                # Normals tensor needs explicit device/dtype
                n_bottom = torch.tensor([0.0, -1.0], device=device, dtype=dtype)
                n_top = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
                n_left = torch.tensor([-1.0, 0.0], device=device, dtype=dtype)
                n_right = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

                # Assign outward normals based on priority
                n_boundary[on_bottom] = n_bottom
                n_boundary[on_top] = n_top
                n_boundary[on_left] = n_left  # Overwrites y at left corners
                n_boundary[on_right] = n_right  # Overwrites y at right corners

                # Apply fixed magnitude (scalar)
                F_boundary = max_magnitude * n_boundary
                F_out_bound_section[idx_on_boundary_section] = F_boundary

            F[idx_out_bound] = F_out_bound_section

    return F


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    box_x_min, box_x_max = -0.1, 0.1
    box_y_min, box_y_max = 0.0, 0.5
    max_magnitude = 0.1
    exp_decay_rate = 10

    print(f"Box defined by: x=[{box_x_min}, {box_x_max}], y=[{box_y_min}, {box_y_max}]")
    print(f"Max Magnitude: {max_magnitude}")
    print(f"Decay: Rate={exp_decay_rate}\n")

    grid_density = 30
    margin = 0.5
    x_range = torch.linspace(box_x_min - margin, box_x_max + margin, grid_density)
    y_range = torch.linspace(box_y_min - margin, box_y_max + margin, grid_density)
    X, Y = torch.meshgrid(x_range, y_range, indexing="xy")

    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    F_vectors_batch = collision_guide(
        points,
        box_x_min,
        box_x_max,
        box_y_min,
        box_y_max,
        max_magnitude=max_magnitude,
        decay_rate=exp_decay_rate,
    )

    # --- Plotting ---
    print("Plotting results...")
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    F_vectors_np = F_vectors_batch.cpu().numpy()
    U = F_vectors_np[:, 0].reshape(X_np.shape)
    V = F_vectors_np[:, 1].reshape(Y_np.shape)

    fig, ax = plt.subplots(figsize=(10, 7))

    magnitude = np.sqrt(U**2 + V**2)
    quiver = ax.quiver(
        X_np,
        Y_np,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="dodgerblue",
        width=0.0035,
    )

    box_patch = Rectangle(
        (box_x_min, box_y_min),
        box_x_max - box_x_min,
        box_y_max - box_y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(box_patch)

    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Vector Field")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(X_np.min(), X_np.max())
    ax.set_ylim(Y_np.min(), Y_np.max())
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()
