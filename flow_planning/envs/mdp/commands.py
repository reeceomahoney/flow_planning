# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import MISSING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
)


class ScheduledPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: ScheduledPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ScheduledPoseCommand, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)  # type: ignore

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]  # type: ignore
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]  # type: ignore

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 4] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.current_stage = 0
        self.cycle_envs = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = (
            combine_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.pose_command_b[:, :3],
                self.pose_command_b[:, 3:],
            )
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        for env_id in env_ids:
            if self.cycle_envs[env_id]:
                # Cycle through commands
                fixed_command = self.cfg.fixed_commands[self.current_stage]
                fixed_command = torch.tensor(fixed_command, device=self.device)

                # add noise
                # position_range = (0.9, 1.1)
                # fixed_command *= math_utils.sample_uniform(
                #     *position_range, fixed_command.shape, fixed_command.device
                # )

                self.pose_command_b[env_id, 0] = fixed_command[0]
                self.pose_command_b[env_id, 1] = fixed_command[1]
                self.pose_command_b[env_id, 2] = fixed_command[2]
            else:
                # Use only the final command
                final_command = self.cfg.fixed_commands[-1]
                self.pose_command_b[env_id, 0] = final_command[0]
                self.pose_command_b[env_id, 1] = final_command[1]
                self.pose_command_b[env_id, 2] = final_command[2]

        # Increment the current stage and wrap around for cycling envs
        self.current_stage = (self.current_stage + 1) % len(self.cfg.fixed_commands)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:]
        )
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class ScheduledPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = ScheduledPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    fixed_commands: list[tuple[float, float, float]] = MISSING
    """List of fixed (x, y, z) commands to cycle through."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/goal_pose"
    )
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
