import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.utils import configclass

from . import mdp
from .franka import T_MAX, FrankaEnvCfg


@configclass
class FrankaFlowPlanningEnvCfg(FrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_fixed, mode="reset"
        )
        self.commands.ee_pose = mdp.FixedPoseCommandCfg(  # type: ignore
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(T_MAX, T_MAX),
            debug_vis=False,
            fixed_commands=[(0.5, 0.3, 0.2)],
        )

        self.scene.obstacle = RigidObjectCfg(  # type: ignore
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.SphereCfg(
                radius=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    roughness=0.1,
                ),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.2)),
        )
