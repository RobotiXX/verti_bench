import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from typing import Callable, List, Any, Optional, Sequence, Type
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import multiprocessing as mp
import torch as th
import numpy as np
import shutil

from verti_bench.envs.wheeled.off_road_VertiBench import off_road_art
from verti_bench.train.custom_networks.artCustomCNN import CustomCombinedExtractor

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.old_episode_num = 0
        self.old_timeout_count = 0
        self.old_fallen_count = 0
        self.old_success_count = 0
        self.old_crash_count = 0
        self.last_ep_rew_mean = 0.0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        return True
    
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Aggregate data from all environments
        """
        total_success_count = sum(self.training_env.get_attr("m_success_count"))
        total_fallen_count = sum(self.training_env.get_attr("m_fallen_count"))
        total_timeout_count = sum(self.training_env.get_attr("m_timeout_count"))
        total_episode_num = sum(self.training_env.get_attr("m_episode_num"))
        total_crash_count = sum(self.training_env.get_attr("m_crash_count"))

        # Log the rates
        self.logger.record("rollout/total_success", total_success_count)
        self.logger.record("rollout/total_fallen", total_fallen_count)
        self.logger.record("rollout/total_timeout", total_timeout_count)
        self.logger.record("rollout/total_episode_num", total_episode_num)
        self.logger.record("rollout/total_crashes", total_crash_count)
        
        self.old_episode_num = total_episode_num
        self.old_timeout_count = total_timeout_count
        self.old_fallen_count = total_fallen_count
        self.old_success_count = total_success_count
        self.old_crash_count = total_crash_count
        
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.last_ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

        return True

    def _on_training_end(self) -> None:
        print("Training ended")
        print("Total episodes ran: ", self.old_episode_num)
        print("Total success count: ", self.old_success_count)
        print("Total fallen count: ", self.old_fallen_count)
        print("Total timeout count: ", self.old_timeout_count)
        print("Total crash count: ", self.old_crash_count)
        return True

def make_env(rank: int = 0, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param stage: (int) the terrain stage
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = off_road_art()
        env.reset(seed=seed + rank)
        return env
        
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/tmp")
    if os.path.exists(terrain_dir):
        shutil.rmtree(terrain_dir)
    
    # Maximum num is 32
    num_procs = 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    base_log_path = "./res/logs/grass"
    os.makedirs(base_log_path , exist_ok=True)

    n_steps = 7000
    num_updates = 15
    timesteps_per_iteration = num_updates * n_steps * num_procs
    
    checkpoint_dir = './res/models/grass'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Vectorized environment
    env = make_vec_env(env_id=make_env(), n_envs=num_procs, vec_env_cls=SubprocVecEnv)
    
    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor,
        # features_extractor_kwargs={'features_dim': 32},
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    new_logger = configure(base_log_path, ["stdout", "csv", "tensorboard"])
    model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=n_steps, batch_size=n_steps // 2, 
                policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=base_log_path, device=device)
    print(model.policy)
    model.set_logger(new_logger)   

    # If training is interrupted, set the checkpoint file
    model = PPO.load(os.path.join(checkpoint_dir, f"ppo_checkpoint1"), env)

    num_iterations = 40
    for i in range(2, num_iterations):  
        model.learn(timesteps_per_iteration, progress_bar=True, callback=TensorboardCallback())
        model.save(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"))
        th.save(model.policy.state_dict(), os.path.join(checkpoint_dir, f"ppo_checkpoint{i}.pt"))
        model = PPO.load(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"), env)  

    print("Training completed!")
    