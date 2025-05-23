from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import mdp
from .franka import FrankaEnvCfg

CONTROLLER = "osc"  # "ik" or "osc"


@configclass
class FrankaRecordEnvCfg(FrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if CONTROLLER == "ik":
            self.actions.arm_action = mdp.CustomIKActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.CustomIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.107)
                ),
            )
        elif CONTROLLER == "osc":
            self.scene.robot.spawn.rigid_props.disable_gravity = False
            self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
            self.scene.robot.actuators["panda_shoulder"].damping = 0.0
            self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
            self.scene.robot.actuators["panda_forearm"].damping = 0.0

            self.actions.arm_action = mdp.CustomOscActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller_cfg=OperationalSpaceControllerCfg(
                    target_types=["pose_abs"],
                    impedance_mode="fixed",
                    inertial_dynamics_decoupling=True,
                    partial_inertial_dynamics_decoupling=False,
                    gravity_compensation=True,
                    motion_stiffness_task=2.5,
                    motion_damping_ratio_task=1.0,
                    nullspace_control="position",
                ),
                nullspace_joint_pos_target="default",
                body_offset=mdp.CustomOscActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )
