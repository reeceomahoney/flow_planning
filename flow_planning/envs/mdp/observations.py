import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat


def ee_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset ee position and orientation in the base frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze() - asset.data.root_pos_w
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7].squeeze()
    rot_mat = matrix_from_quat(quat)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([pos, ortho6d], dim=-1)


def ee_pose_l(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset ee position and orientation in the env frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze() - env.scene.env_origins
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7].squeeze()
    rot_mat = matrix_from_quat(quat)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([pos, ortho6d], dim=-1)


def position_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, :3]
