import math

import pytorch_kinematics as pk
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .commands import FixedPoseCommand


def reset_joints_uniform_task_space(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    lower_limits: tuple[float, float],
    upper_limits: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to a random position in task space."""
    # extract joint info
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids].clone()
    # task space limits
    lower = torch.tensor(lower_limits, device=env.device)
    upper = torch.tensor(upper_limits, device=env.device)

    # build urdf chain for ik
    chain = pk.build_serial_chain_from_urdf(
        open("data/urdf/panda.urdf", mode="rb").read(), "panda_hand"
    ).to(device=env.device)

    # get target position and orientation
    ee_pos = math_utils.sample_uniform(
        lower, upper, (env_ids.shape[0], 3), str(env.device)
    )
    ee_ori = torch.tensor([[0.0, math.pi, 0.0]], device=env.device).expand(
        env_ids.shape[0], -1
    )
    goal_tf = pk.Transform3d(pos=ee_pos, rot=ee_ori, device=str(env.device))

    # solve ik
    ik = pk.PseudoInverseIK(
        chain,
        max_iterations=30,
        retry_configs=joint_pos[0:1, :7],
        joint_limits=joint_pos_limits[0].T,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="all",
        debug=False,
        lr=0.2,
    )
    sol = ik.solve(goal_tf)

    # set into the physics simulation
    joint_pos[:, :7] = sol.solutions.squeeze(1)
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,  # type: ignore
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

    # fmt: off
    pos_list = [
        # (0.5, -0.3, 0.2)
        [-1.6615e-01, 2.7841e-01, -3.8028e-01, -2.0778e00, 1.3647e-01, 2.3238e00, 1.4746e-01, 0, 0],
        # (0.5, -0.3, 0.6)
        [-2.7545e-01, 2.4703e-01, -3.3734e-01, -1.0436e00, 8.1770e-02, 1.2697e00, 1.9731e-01, 0, 0],
    ]
    # fmt: on

    # build joint position
    joint_pos = torch.tensor(pos_list[idx], device=env.device)
    joint_pos = joint_pos.unsqueeze(0).expand(env_ids.shape[0], -1)

    # set velocities to zero
    joint_vel = asset.data.default_joint_vel[env_ids]
    joint_vel = torch.zeros_like(joint_vel)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,  # type: ignore
    )
