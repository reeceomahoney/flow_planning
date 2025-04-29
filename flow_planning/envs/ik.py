from flow_planning.envs.mdp.actions import CustomOscActionCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .rsl_rl import FRANKA_PANDA_NO_PD_CFG, T_MAX, FrankaRLEnvCfg


@configclass
class FrankaIKEnvCfg(FrankaRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        controller = "osc"
        if controller == "ik":
            self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )  # type: ignore
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
        elif controller == "osc":
            self.scene.robot = FRANKA_PANDA_NO_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )  # type: ignore
            self.actions.arm_action = CustomOscActionCfg(
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
                body_offset=CustomOscActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )
        # self.episode_length_s = 2 * T_MAX
