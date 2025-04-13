from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .rsl_rl import FrankaRLEnvCfg


@configclass
class FrankaIKEnvCfg(FrankaRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
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
