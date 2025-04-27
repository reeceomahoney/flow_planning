"""This file is needed for us to define a custom IK controller with some extra action processing"""

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.task_space_actions import (
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
