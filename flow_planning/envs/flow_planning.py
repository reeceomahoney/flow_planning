import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .rsl_rl import T_MAX, FrankaRLEnvCfg


@configclass
class FrankaFlowPlanningEnvCfg(FrankaRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.observations.policy.pose_command = None  # type: ignore
        self.curriculum = None  # type: ignore
        self.events.reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_fixed, mode="reset"
        )
        self.events.apply_random_force = None  # type: ignore
        self.commands.ee_pose = mdp.FixedPoseCommandCfg(  # type: ignore
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(T_MAX, T_MAX),
            debug_vis=True,
            fixed_commands=[(0.7, 0, 0.2)],
        )


@configclass
class FrankaGuidanceEnvCfg(FrankaFlowPlanningEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle = RigidObjectCfg(  # type: ignore
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.8, 0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1.0,
                    max_angular_velocity=1.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.3)),
        )
