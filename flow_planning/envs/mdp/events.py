import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .commands import FixedPoseCommand


def reset_joints_random(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Reset the robot joints to a random position within its full range."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state for shape
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # sample position
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = math_utils.sample_uniform(
        joint_pos_limits[..., 0],
        joint_pos_limits[..., 1],
        joint_pos.shape,
        str(joint_pos.device),
    )

    # set velocities to zero
    joint_vel = torch.zeros_like(joint_vel)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids.tolist()
    )


def reset_joints_fixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints from a fixed list of positions."""
    # get asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get command stage
    cmd_manager = env.command_manager.get_term("ee_pose")
    assert isinstance(cmd_manager, FixedPoseCommand)
    idx = cmd_manager.current_stage

    rel_pos_list = [
        # joint pos for (0.5, -0.3, 0.2)
        [0.1805, 0.9034, -0.7486, 0.7486, 0.2993, -0.7417, -0.7022, -0.0473, -0.0339],
        # joint pos for (0.5, -0.3, 0.6)
        [-0.0067, 0.7955, -0.6834, 1.6919, 0.1493, -1.7369, -0.5888, -0.0330, -0.0499],
    ]

    # build joint position
    joint_pos_rel = torch.tensor(rel_pos_list[idx], device=env.device)
    joint_pos_rel = joint_pos_rel.unsqueeze(0).expand(env_ids.shape[0], -1)
    joint_pos = joint_pos_rel + asset.data.default_joint_pos[env_ids]

    # set velocities to zero
    joint_vel = asset.data.default_joint_vel[env_ids]
    joint_vel = torch.zeros_like(joint_vel)

    # add noise
    # position_range = (0.95, 1.05)
    # velocity_range = (0.0, 0.0)
    # joint_pos *= math_utils.sample_uniform(
    #     *position_range, joint_pos.shape, joint_pos.device
    # )
    # joint_vel *= math_utils.sample_uniform(
    #     *velocity_range, joint_vel.shape, joint_vel.device
    # )

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,  # type: ignore
    )
