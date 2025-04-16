import logging
import os
import statistics
import time
from collections import deque

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from rsl_rl.utils import store_code_state
from tqdm import tqdm, trange

from flow_planning.envs import ParticleEnv
from flow_planning.policy import ClassifierPolicy, Policy
from flow_planning.utils import (
    ExponentialMovingAverage,
    InferenceContext,
    Normalizer,
    get_dataloaders,
    get_goal,
)
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper

# A logger for this file
log = logging.getLogger(__name__)


def process_ep_data(x, new_ids):
    return x[new_ids][:, 0].detach().cpu().numpy().tolist()


class Runner:
    def __init__(
        self,
        env: RslRlVecEnvWrapper | ParticleEnv,
        agent_cfg,
        log_dir: str | None = None,
        device="cpu",
    ):
        self.env = env
        # self.env.reset()
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader = get_dataloaders(**self.cfg.dataset)
        self._create_policy()

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.parameters(), self.cfg.ema_decay, self.cfg.device
        )
        self.use_ema = agent_cfg.use_ema

        # variables
        if isinstance(env, RslRlVecEnvWrapper | ParticleEnv):
            self.num_steps_per_env = self.env.max_episode_length  # type: ignore
        self.log_dir = log_dir
        self.current_learning_iteration = 0

        # logging
        if self.log_dir is not None:
            # initialize wandb
            wandb.config = OmegaConf.to_container(
                agent_cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                project=self.cfg.experiment.wandb_project,
                dir=log_dir,
                config=wandb.config,  # type: ignore
            )
            # make model directory
            os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)  # type: ignore
            # save git diffs
            store_code_state(self.log_dir, [__file__])

    def _create_policy(self):
        model = hydra.utils.instantiate(self.cfg.model)
        normalizer = Normalizer(self.train_loader, self.cfg.scaling, self.device)
        self.policy = Policy(model, normalizer, self.env, **self.cfg.policy)

    def learn(self):
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
        self.policy.train()

        rewbuffer = deque()
        lenbuffer = deque()
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter, dynamic_ncols=True):
            start = time.time()

            # simulation
            if it % self.cfg.sim_interval == 0:
                t = 0
                ep_infos = []
                self.env.reset()

                with InferenceContext(self) and tqdm(
                    total=self.num_steps_per_env, desc="Simulating...", leave=False
                ) as pbar:
                    while t < self.num_steps_per_env:
                        goal = get_goal(self.env)
                        actions = self.policy.act({"obs": obs, "goal": goal})["action"]

                        # step the environment
                        for i in range(self.policy.T_action):
                            obs, rewards, dones, infos = self.env.step(actions[:, i])

                            if t == self.num_steps_per_env - 1:
                                dones = torch.ones_like(dones)

                            # move device
                            obs, rewards, dones = (
                                obs.to(self.device),
                                rewards.to(self.device),
                                dones.to(self.device),
                            )

                            if self.log_dir is not None:
                                # rewards and dones
                                if "log" in infos:
                                    ep_infos.append(infos["log"])
                                cur_reward_sum += rewards
                                cur_episode_length += 1
                                new_ids = (dones > 0).nonzero(as_tuple=False)
                                rewbuffer.extend(
                                    process_ep_data(cur_reward_sum, new_ids)
                                )
                                lenbuffer.extend(
                                    process_ep_data(cur_episode_length, new_ids)
                                )
                                cur_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0

                            t += 1
                            pbar.update(1)

            # evaluation
            if it % self.cfg.eval_interval == 0:
                with InferenceContext(self):
                    test_mse, test_obs_mse, test_act_mse = [], [], []
                    for batch in tqdm(self.test_loader, desc="Testing...", leave=False):
                        mse, obs_mse, act_mse = self.policy.test(batch)
                        test_mse.append(mse)
                        test_obs_mse.append(obs_mse)
                        test_act_mse.append(act_mse)
                    test_mse = statistics.mean(test_mse)
                    test_obs_mse = statistics.mean(test_obs_mse)
                    test_act_mse = statistics.mean(test_act_mse)

                    self.policy.plot(it)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss = self.policy.update(batch)
            self.ema_helper.update(self.policy.parameters())

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", f"model_{it}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "models", "model.pt"))

    def log(self, locs: dict):
        # training
        wandb.log(
            {
                "Loss/loss": locs["loss"],
                "Perf/iter_time": locs["iter_time"] / self.cfg.log_interval,
            },
            step=locs["it"],
        )
        # evaluation
        if locs["it"] % self.cfg.eval_interval == 0:
            wandb.log(
                {
                    "Loss/test_mse": locs["test_mse"],
                    "Loss/test_obs_mse": locs["test_obs_mse"],
                    "Loss/test_act_mse": locs["test_act_mse"],
                },
                step=locs["it"],
            )
        # simulation
        if locs["it"] % self.cfg.sim_interval == 0:
            if locs["ep_infos"]:
                for key in locs["ep_infos"][0]:
                    # get the mean of each ep info value
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs["ep_infos"]:
                        # handle scalar and zero dimensional tensor infos
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat(
                            (infotensor, ep_info[key].to(self.device))
                        )
                    value = torch.mean(infotensor)
                    # log
                    if "/" in key:
                        wandb.log({key: value}, step=locs["it"])
                    else:
                        wandb.log({"Episode/" + key, value}, step=locs["it"])
            wandb.log(
                {
                    "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                    "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
                },
                step=locs["it"],
            )

    def save(self, path):
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())

        saved_dict = {
            "model_state_dict": self.policy.model.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.policy.normalizer.state_dict(),
            "iter": self.current_learning_iteration,
        }
        torch.save(saved_dict, path)

        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device)
        self.policy.model.load_state_dict(loaded_dict["model_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.policy.normalizer.load_state_dict(loaded_dict["norm_state_dict"])


class ClassifierRunner(Runner):
    def _create_policy(self):
        model = hydra.utils.instantiate(self.cfg.model)
        normalizer = Normalizer(self.train_loader, self.cfg.scaling, self.device)
        self.policy = ClassifierPolicy(model, normalizer, self.env, **self.cfg.policy)

    def learn(self):
        self.policy.train()

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter, dynamic_ncols=True):
            start = time.time()

            # evaluation
            if it % self.cfg.eval_interval == 0:
                test_mse = []
                for batch in tqdm(self.test_loader, desc="Testing...", leave=False):
                    mse = self.policy.test(batch)
                    test_mse.append(mse)
                test_mse = statistics.mean(test_mse)

                self.policy.plot(it)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss = self.policy.update(batch)
            self.ema_helper.update(self.policy.parameters())

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.eval_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", "model.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "models", "model.pt"))

    def log(self, locs: dict):
        # training
        wandb.log(
            {
                "Loss/loss": locs["loss"],
                "Perf/iter_time": locs["iter_time"] / self.cfg.log_interval,
            },
            step=locs["it"],
        )
        # evaluation
        if locs["it"] % self.cfg.eval_interval == 0:
            wandb.log(
                {"Loss/test_mse": locs["test_mse"]},
                step=locs["it"],
            )

    def save(self, path):
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())

        saved_dict = {
            "model_state_dict": self.policy.model.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.policy.normalizer.state_dict(),
            "classifier_state_dict": self.policy.classifier.state_dict(),
            "iter": self.current_learning_iteration,
        }
        torch.save(saved_dict, path)

        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device)
        self.policy.model.load_state_dict(loaded_dict["model_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.policy.normalizer.load_state_dict(loaded_dict["norm_state_dict"])
        self.policy.classifier.load_state_dict(loaded_dict["classifier_state_dict"])

    def load_model(self, path):
        loaded_dict = torch.load(path, map_location=self.device)
        self.policy.model.load_state_dict(loaded_dict["model_state_dict"])
