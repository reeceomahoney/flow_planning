"""This file is needed for us to define a custom IK controller with some extra action processing"""

import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)
from isaaclab.utils import configclass


@configclass
class CustomOscActionCfg(actions_cfg.OperationalSpaceControllerActionCfg):
    def __post_init__(self):
        self.class_type = CustomOscAction


class CustomOscAction(OperationalSpaceControllerAction):
    def __init__(
        self,
        cfg: CustomOscActionCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)
        self.noise_stds = torch.tensor(
            [1, 1, 1, 1, 0.1, 0.1, 0.1], device=self.device
        ).repeat(self.num_envs, 1)

    def apply_actions(self):
        """Computes the joint efforts for operational space control and applies them to the articulation."""

        # Update the relevant states and dynamical quantities
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()
        # Calculate the joint efforts
        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )
        # Add noise
        self._joint_efforts += torch.normal(mean=0.0, std=self.noise_stds)
        # Apply the joint efforts
        self._asset.set_joint_effort_target(
            self._joint_efforts, joint_ids=self._joint_ids
        )


class CustomIKController(DifferentialIKController):
    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        joint_pos_des = super().compute(ee_pos, ee_quat, jacobian, joint_pos)
        # apply the custom scaling factor
        delta_joint_pos = joint_pos_des - joint_pos
        joint_pos_des = joint_pos + 0.15 * delta_joint_pos
        joint_pos_des += 0.2 * torch.randn_like(joint_pos_des)
        return joint_pos_des


class CustomIKAction(DifferentialInverseKinematicsAction):
    def __init__(
        self,
        cfg: actions_cfg.DifferentialInverseKinematicsActionCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)
        self._ik_controller = CustomIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self.joint_pos_des = joint_pos_des
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class CustomIKControllerCfg(DifferentialIKControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.class_type = CustomIKController


@configclass
class CustomIKActionCfg(actions_cfg.DifferentialInverseKinematicsActionCfg):
    def __post_init__(self):
        self.class_type = CustomIKAction
