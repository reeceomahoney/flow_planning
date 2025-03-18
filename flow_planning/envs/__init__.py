import gymnasium as gym

from . import flow_planning, rsl_rl

gym.register(
    id="Isaac-Franka-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rsl_rl.FrankaRLEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-FlowPlanning",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flow_planning.FrankaFlowPlanningEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-Guidance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flow_planning.FrankaGuidanceEnvCfg,
    },
)
