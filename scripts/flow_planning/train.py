from isaaclab.app import AppLauncher

# NOTE: We need to run the app launcher first to avoid import errors
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flow_planning.envs  # noqa: F401
from flow_planning.runner import ClassifierRunner, Runner
from flow_planning.utils import create_env
from isaaclab.utils.io import dump_pickle, dump_yaml

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../config/flow_planning", config_name="cfg.yaml"
)
def main(agent_cfg: DictConfig):
    # set log dir
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Logging experiment in directory: {log_dir}")

    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    # create environment
    env_name = agent_cfg.env.env_name
    experiment = agent_cfg.experiment.wandb_project
    env, agent_cfg, env_cfg = create_env(env_name, agent_cfg)

    # create runner
    if experiment == "classifier":
        runner = ClassifierRunner(
            env, agent_cfg, log_dir=log_dir, device=agent_cfg.device
        )
        model_path = "/home/reece/ws/dev/flow_planning/logs/flow_planning/Apr-17/22-54-36/models/model_50000.pt"
        runner.load_model(model_path)
    else:
        runner = Runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)

    # dump the configuration into log-directory
    agent_cfg_dict = OmegaConf.to_container(agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg_dict)
    if env_name.startswith("Isaac"):
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    runner.learn()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
