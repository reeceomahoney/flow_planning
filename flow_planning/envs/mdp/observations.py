import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def ee_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset ee position and orientation in the base frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain quantities from simulation
    body_idx = asset_cfg.body_ids[0]  # type: ignore
    ee_pos_w = asset.data.body_pos_w[:, body_idx]
    ee_quat_w = asset.data.body_quat_w[:, body_idx]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )

    # account for the offset
    offset_pos = torch.tensor([0.0, 0.0, 0.107], device=env.device).repeat(
        env.num_envs, 1
    )
    offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(
        env.num_envs, 1
    )
    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        ee_pose_b, ee_quat_b, offset_pos, offset_rot
    )

    # convert to the ortho6d representation
    rot_mat = math_utils.matrix_from_quat(ee_quat_b)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([ee_pose_b, ortho6d], dim=-1)
