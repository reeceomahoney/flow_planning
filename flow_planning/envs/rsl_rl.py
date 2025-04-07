import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ReachEnvCfg,
)

from . import mdp

T_MAX = 64 / 30

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(T_MAX, T_MAX),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 1),
            pos_y=(-0.5, 0.5),
            pos_z=(0.15, 0.9),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )

    # ee_pose = mdp.ScheduledPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="panda_hand",
    #     resampling_time_range=(T_MAX, T_MAX),
    #     debug_vis=True,
    #     fixed_commands=[(0.8, 0, 0.6), (0.8, 0, 0.2)],
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee_pose = ObsTerm(
            func=mdp.ee_pose,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")},
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    # reset_robot_joints = EventTermCfg(
    #     func=mdp.reset_joints_fixed,
    #     mode="reset",
    # )

    # apply_random_force = EventTermCfg(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.3, 0.5),
    #     params={
    #         "force_range": (-20, 20),
    #         "torque_range": (0, 0),
    #         "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
    #     },
    # )


##
# Environment configuration
##


@configclass
class FrankaRLEnvCfg(ReachEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "panda_hand"
        ]
        self.rewards.end_effector_position_tracking_fine_grained.params[
            "asset_cfg"
        ].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["panda_hand"]

        # curriculum
        self.curriculum.action_rate.params["weight"] = -0.05
        self.curriculum.joint_vel.params["weight"] = -0.01

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # general settings
        self.episode_length_s = T_MAX
