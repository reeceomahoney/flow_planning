import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_joints_random(
    env: ManagerBasedEnv,
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
        joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def reset_joints_fixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints from a fixed list of positions.

    ee (x,z)   | joint_pos
    -------------------------
    (0.4, 0.2) | [-2.2671e-02,  7.6667e-02,  2.1580e-02, -2.7259e+00, -4.9747e-03, 2.8028e+00,  7.8894e-01,  9.8029e-05,  1.0173e-04]
    (0.6, 0.6) | [-1.6977e-02,  1.3992e-01,  1.8799e-02, -1.4575e+00, -2.6461e-03, 1.5982e+00,  7.8706e-01,  9.3350e-05,  1.0834e-04]
    (0.4, 0.6) | [-1.7655e-02, -4.9210e-01,  1.3407e-02, -2.1302e+00,  6.3447e-03, 1.6397e+00,  7.7912e-01,  9.3380e-05,  1.0829e-04]
    (0.6, 0.2) | [-2.9749e-02,  4.9310e-01,  2.9753e-02, -2.0311e+00, -2.4329e-02, 2.5243e+00,  8.0171e-01,  9.6156e-05,  1.0438e-04]
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    position_range = (0.8, 1.2)
    velocity_range = (0.0, 0.0)

    pos_list = [
        [
            -2.2671e-02,
            7.6667e-02,
            2.1580e-02,
            -2.7259e00,
            -4.9747e-03,
            2.8028e00,
            7.8894e-01,
            9.8029e-05,
            1.0173e-04,
        ],
        [
            -1.7655e-02,
            -4.9210e-01,
            1.3407e-02,
            -2.1302e00,
            6.3447e-03,
            1.6397e00,
            7.7912e-01,
            9.3380e-05,
            1.0829e-04,
        ],
    ]
    idx = env.command_manager._terms["ee_pose"].current_stage
    joint_pos = (
        torch.tensor(pos_list[idx])
        .unsqueeze(0)
        .expand(env_ids.shape[0], -1)
        .to(env_ids.device)
    )

    # set velocities to zero
    joint_vel = asset.data.default_joint_vel[env_ids]
    joint_vel = torch.zeros_like(joint_vel)

    # add noise
    # joint_pos *= math_utils.sample_uniform(
    #     *position_range, joint_pos.shape, joint_pos.device
    # )
    # joint_vel *= math_utils.sample_uniform(
    #     *velocity_range, joint_vel.shape, joint_vel.device
    # )

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
