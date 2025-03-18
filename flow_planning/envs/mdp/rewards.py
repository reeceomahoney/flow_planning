import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_error_magnitude,
    quat_mul,
)


def ee_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.exp(-((torch.norm(curr_pos_w - des_pos_w, dim=1) / std) ** 2))


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return torch.exp(-quat_error_magnitude(curr_quat_w, des_quat_w))


def ee_9d_pose_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 1,
) -> torch.Tensor:
    """Penalize tracking of the end-effector position and orentation error."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # current ee position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze(1)
    # current ee ortho6d
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7]
    rot_mat = matrix_from_quat(quat)
    curr_ortho6d_w = rot_mat[..., :2].reshape(-1, 6)

    # desired ee position
    des_pos_b = command[:, :3]
    des_pos_w = des_pos_b + env.scene.env_origins
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    # desired ee ortho6d
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    des_rot_mat = matrix_from_quat(des_quat_w)
    des_ortho6d_w = des_rot_mat[..., :2].reshape(-1, 6)

    # compute the error
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    orientation_error = torch.norm(curr_ortho6d_w - des_ortho6d_w, dim=1)

    reward = torch.exp(-(pos_error / std + orientation_error / 4 * std))
    return reward
