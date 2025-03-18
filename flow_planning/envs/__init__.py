import gymnasium as gym

from .flow_planning import FrankaFlowPlanningEnvCfg, FrankaGuidanceEnvCfg
from .maze import MazeEnv  # noqa: F401
from .particle import ParticleEnv  # noqa: F401
from .rsl_rl import FrankaRLEnvCfg

gym.register(
    id="Isaac-Franka-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaRLEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-FlowPlanning",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaFlowPlanningEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-Guidance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaGuidanceEnvCfg,
    },
)
