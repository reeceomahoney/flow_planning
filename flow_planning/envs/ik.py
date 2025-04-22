from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp
from .rsl_rl import FrankaRLEnvCfg


@configclass
class FrankaIKEnvCfg(FrankaRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot"
        # )
        # self.actions.arm_action = mdp.CustomIKActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     body_name="panda_hand",
        #     controller=mdp.CustomIKControllerCfg(
        #         command_type="pose", use_relative_mode=False, ik_method="dls"
        #     ),
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
        #         pos=(0.0, 0.0, 0.107)
        #     ),
        # )

        FRANKA_PANDA_CFG.actuators["panda_shoulder"].stiffness = 0.0
        FRANKA_PANDA_CFG.actuators["panda_shoulder"].damping = 0.0
        FRANKA_PANDA_CFG.actuators["panda_forearm"].stiffness = 0.0
        FRANKA_PANDA_CFG.actuators["panda_forearm"].damping = 0.0
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
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
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.107)
            ),
        )
