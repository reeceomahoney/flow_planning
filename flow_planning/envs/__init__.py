import gymnasium as gym

from .flow_planning import FrankaFlowPlanningEnvCfg
from .ik import FrankaRecordEnvCfg
from .particle import ParticleEnv  # noqa: F401

gym.register(
    id="Isaac-Franka-Record",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaRecordEnvCfg,
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
