import gymnasium as gym
from stable_baselines3 import PPO
from verti_bench.envs.wheeled.off_road_VertiBench_ACL import off_road_art
from gymnasium.utils.env_checker import check_env

import numpy as np
import os
import shutil
import torch

render = True
if __name__ == '__main__':
    terrain_dir1 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/tmp")
    terrain_dir2 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/tmp")
    if os.path.exists(terrain_dir1):
        shutil.rmtree(terrain_dir1)
    if os.path.exists(terrain_dir2):
        shutil.rmtree(terrain_dir2)
    
    env = off_road_art(world_id=5, pos_id=1)
    checkpoint_dir = './VertBenchModels'
    loaded_model = PPO.load(os.path.join(checkpoint_dir, f"ppo_ACL_iter102_level18"), env)

    #Render and test model
    totalSteps = 7000
    obs, _ = env.reset()
    if render:
        env.render('follow')
    for step in range(totalSteps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", (terminated or truncated))
        if render:
            env.render('follow')
        if (terminated or truncated):
            break
