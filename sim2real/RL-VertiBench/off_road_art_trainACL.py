import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from typing import Callable, List, Any, Optional, Sequence, Type
import os
from collections import defaultdict

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
import gc
import pickle

from verti_bench.envs.wheeled.off_road_artACL import off_road_art
from verti_bench.train.custom_networks.artCustomCNN import CustomCombinedExtractor
from verti_bench.envs.utils.SafeSubproc_vec_env import SafeSubprocVecEnv

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
        A rollout means filling up the RolloutBuffer with buffer_size transitions.
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
    
class ScoreUpdateCallback(BaseCallback):
    def __init__(self, num_levels, verbose=0):
        super(ScoreUpdateCallback, self).__init__(verbose)
        self.level_scores = np.zeros(num_levels)
        self.level_staleness = np.zeros(num_levels)
        self.unseen_levels = np.ones(num_levels)
        self.level_returns = defaultdict(list)
        self.level_value_preds = defaultdict(list)
        self.level_dones = defaultdict(list)
        self.current_level_index = None
        
        # Parameters for Prioritized Level Replay
        self.temperature = 0.1
        self.staleness_coef = 0.1
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        Access the current environment state
        """
        for env_idx, info in enumerate(self.locals['infos']):
            level_index = info['level_index']
            reward = self.locals['rewards'][env_idx]
            value_pred = self.locals['values'][env_idx].cpu().numpy()
            done = self.locals['dones'][env_idx]

            self.level_returns[level_index].append(reward)
            self.level_value_preds[level_index].append(value_pred)
            self.level_dones[level_index].append(done)

            if done:
                self.update_level_score(level_index, partial=False)

        return True

    def _on_rollout_end(self) -> None:
        for level_index in list(self.level_returns.keys()):
            # Check if the level reached termination
            if any(self.level_dones[level_index]):
                self.update_level_score(level_index, partial=False)
            else:
                self.update_level_score(level_index, partial=True)

        # Update staleness
        self.update_staleness()
        
    def update_level_score(self, level_index, partial=False):
        returns = np.array(self.level_returns[level_index])
        value_preds = np.array(self.level_value_preds[level_index])
        dones = np.array(self.level_dones[level_index])
        
        remaining_returns = []
        remaining_value_preds = []
        remaining_dones = []
        
        if len(returns) > 0:
            # Calculate score only up to the done flag
            if not partial and np.any(dones):
                end_index = np.where(dones)[0][0] + 1
                returns_to_use = returns[:end_index]
                value_preds_to_use = value_preds[:end_index]
                remaining_returns = returns[end_index:]
                remaining_value_preds = value_preds[end_index:]
                remaining_dones = dones[end_index:]
            else:
                returns_to_use = returns
                value_preds_to_use = value_preds
            
            score = self.update_scores(returns_to_use, value_preds_to_use)
            print(f"Level {level_index} score: {score}")
            self.level_scores[level_index] = 0.9 * self.level_scores[level_index] + 0.1 * score
            self.unseen_levels[level_index] = 0

        # If partial, keep the last unfinished part
        if partial:
            self.level_returns[level_index] = list(remaining_returns)
            self.level_value_preds[level_index] = list(remaining_value_preds)
            self.level_dones[level_index] = list(remaining_dones)
        else:
            self.level_returns[level_index].clear()
            self.level_value_preds[level_index].clear()
            self.level_dones[level_index].clear()

    def update_scores(self, returns, value_preds):
        advantages = returns - value_preds
        return np.mean(np.abs(advantages))

    def update_staleness(self):
        self.level_staleness += 1
        if self.current_level_index is not None:
            self.level_staleness[self.current_level_index] = 0

    def sample_level(self):
        num_unseen = np.sum(self.unseen_levels > 0)
        proportion_seen = (len(self.unseen_levels) - num_unseen) / len(self.unseen_levels)
        randNum = np.random.rand()
        
        print(f"Proportion of seen levels: {proportion_seen}")
        print(f"Random number: {randNum}")

        if proportion_seen >= 0.1 and randNum > 0.5:
            return self._sample_replay_level()
        else:
            return self._sample_unseen_level()
        
    def _sample_replay_level(self):
        scores = self.level_scores
        staleness = self.level_staleness

        score_weights = self._score_transform('rank', self.temperature, scores)
        score_weights *= (1 - self.unseen_levels)

        if self.staleness_coef > 0:
            staleness_weights = self._score_transform('rank', self.temperature, staleness)
            staleness_weights *= (1 - self.unseen_levels)
            score_weights = (1 - self.staleness_coef) * score_weights + self.staleness_coef * staleness_weights
            
        score_weights /= np.sum(score_weights)
        self.current_level_index = np.random.choice(np.arange(len(scores)), p=score_weights)
        self.update_staleness()
        
        # Debugging output
        print(f"Sampled replay level index: {self.current_level_index}")
        print(f"Score weights: {score_weights}")
        print(f"Scores: {scores}")
        print(f"Staleness: {staleness}")
        print(f"Unseen levels: {self.unseen_levels}")

        return self.current_level_index
    
    def _sample_unseen_level(self):
        sample_weights = self.unseen_levels / np.sum(self.unseen_levels)
        seed_idx = np.random.choice(np.arange(len(self.unseen_levels)), p=sample_weights)
        self.current_level_index = seed_idx
        self.update_staleness()
        
        # Debugging output
        print(f"Sampled unseen level index: {self.current_level_index}")
        print(f"Unseen weights: {sample_weights}")
        print(f"Unseen levels: {self.unseen_levels}")

        return self.current_level_index
    
    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        elif transform == 'rank':
            temp = np.flip(np.argsort(scores))
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1. / temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1. / temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores) / temperature)
        else:
            raise ValueError(f"Unsupported transform: {transform}")

        return weights
    
    
def make_env(level_index, rank: int = 0, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param stage: (int) the terrain stage
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :return: (Callable)
    """
    def _init() -> gym.Env:
        try:
            env = off_road_art()
            env.update_terrain_stage(level_index)
            env.set_nice_vehicle_mesh()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            
            original_step = env.step
            def wrapped_step(action):
                obs, reward, done, truncated, info = original_step(action)
                info['level_index'] = level_index
                return obs, reward, done, truncated, info
            
            env.step = wrapped_step
            return env
        except Exception as e:
            print(f"Failed to initialize environment in subprocess {rank} with seed {seed}: {str(e)}")
            raise e

    return _init


if __name__ == '__main__':
    set_random_seed(0)
    
    num_levels = 100
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Maximum num is 32
    num_procs = 20                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    log_path = "./logs/vws_logs_PLR2/"
    os.makedirs(log_path, exist_ok=True)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs={'features_dim': 32},
        activation_fn=th.nn.ReLU,
        net_arch=dict(activation_fn=th.nn.ReLU, pi=[64, 64], vf=[64, 64])
        # net_arch=dict(activation_fn=th.nn.ReLU, pi=[64, 32, 16], vf=[64, 32, 16])
    )
    
    # Update policy every num_episodes = num_procs * num_group
    # n_steps = max_time * control_freq * num_group
    n_steps = 20 * 10 * 2
    success_rate_eval = 0

    num_updates = 20 
    total_timesteps = num_updates * n_steps * num_procs
    
    checkpoint_dir = './logs/vws_ppo_checkpoints_PLR2/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    iteration = 200
    start_iter = 0
    checkpoint_file = None
    state_file = None

    # # If training is interrupted, set the checkpoint file
    # checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_iter2_level75")
    # state_file = os.path.join(checkpoint_dir, f"score_iter2_level75.pkl")
    # start_iter = 3
    
    score_callback = ScoreUpdateCallback(num_levels)
    
    if state_file and os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            state_dict = pickle.load(f)
            score_callback.unseen_levels = state_dict['unseen_levels']
            score_callback.level_scores = state_dict['level_scores']
            score_callback.level_staleness = state_dict['level_staleness']
            score_callback.current_level_index = state_dict['current_level_index']
        print(f"Loaded score callback state from {state_file}")
    else:
        print(f"No saved state found, starting fresh")
    
    
    for i in range(start_iter, iteration):
        print(f"Training iter {i+1}/{iteration}")
        
        level_index = score_callback.sample_level()
        print(f"Selected level: {level_index}")
        # Vectorized environment
        env = make_vec_env(env_id=make_env(level_index), n_envs=num_procs, vec_env_cls=SubprocVecEnv)
        
        # Initialize or load the model
        if checkpoint_file:
            model = PPO.load(checkpoint_file, env=env, device=device)
            print(f"Loaded model from {checkpoint_file}")
        else:
            # Initialize the model
            model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=n_steps, batch_size=n_steps // 2,
                    policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path, device=device)
            print(model.policy)
        
        iter_log_path = os.path.join(log_path, f"iter{i}_level{level_index}")
        os.makedirs(iter_log_path, exist_ok=True)
        new_logger = configure(iter_log_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Instantiate the callback
        callback = TensorboardCallback()
        callbacks = [callback, score_callback]
        
        model.learn(total_timesteps, progress_bar=True, callback=callbacks)
        
        checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_iter{i}_level{level_index}")
        model.save(checkpoint_file)
        th.save(model.policy.state_dict(), os.path.join(checkpoint_dir, f"ppo_checkpoint_iter{i}_level{level_index}.pt"))
        del model
        del env
        gc.collect()
        th.cuda.empty_cache()
        
        # Save the scores after each iteration
        with open(f"./{log_path}level_scores.txt", "a") as f:
            f.write(f"Iteration {i+1} Scores:\n")
            for level, score in enumerate(score_callback.level_scores):
                f.write(f"Level {level}: {score}\n")

        with open(f"./{log_path}level_staleness.txt", "a") as f:
            f.write(f"Iteration {i+1} Staleness:\n")
            for level, staleness in enumerate(score_callback.level_staleness):
                f.write(f"Level {level}: {staleness}\n")

        # Save the state of the score callback
        state_dict = {
            'unseen_levels': score_callback.unseen_levels,
            'level_scores': score_callback.level_scores,
            'level_staleness': score_callback.level_staleness,
            'current_level_index': score_callback.current_level_index
        }

        with open(os.path.join(checkpoint_dir, f"score_iter{i}_level{level_index}.pkl"), 'wb') as f:
            pickle.dump(state_dict, f)
