import gymnasium as gym
from stable_baselines3 import PPO
from verti_bench.envs.wheeled.off_road_VertiBench_ACL import off_road_art
from gymnasium.utils.env_checker import check_env

import numpy as np
import os
import shutil
import csv
import json

if __name__ == '__main__':    
    # Create results directory and files
    res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./res/ACL-rebuttal")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, 'ACL_eval.csv')
    json_path = os.path.join(res_dir, 'ACL_eval.json')
    
    # Define CSV fields
    fieldnames = [
        'world_id', 'success_rate', 'mean_traversal_time', 'variance_traversal_time',
        'mean_roll', 'variance_roll', 'mean_pitch', 'variance_pitch',
        'mean_throttle', 'variance_throttle', 'mean_steering', 'variance_steering'
    ]
    
    # Initialize CSV and JSON files
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    with open(json_path, 'w') as jsonfile:
        json.dump([], jsonfile)
    
    for world_id in range(1, 101):
        print(f"Evaluating World {world_id}/{100}")
        
        terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/tmp")
        if os.path.exists(terrain_dir):
            shutil.rmtree(terrain_dir)
    
        world_results = []
        for pos_id in range(10):
            env = off_road_art(world_id=world_id, pos_id=pos_id)
            checkpoint_dir = './VertBenchModels'
            loaded_model = PPO.load(os.path.join(checkpoint_dir, f"ppo_ACL_iter102_level18"), env)
            
            obs, _ = env.reset()
            terminated = truncated = False
            
            # Store episode data
            roll_angles = []
            pitch_angles = []
            throttle_data = []
            steering_data = []
            vehicle_states = []
            
            # Run episode
            while not (terminated or truncated):
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Store data from info dictionary
                roll_angles.extend(info['roll_angles'])
                pitch_angles.extend(info['pitch_angles'])
                throttle_data.extend(info['throttle_data'])
                steering_data.extend(info['steering_data'])
                vehicle_states.extend(info['vehicle_states'])

            # Store episode results
            world_results.append({
                'time_to_goal': info['time_to_goal'] if info['success'] else None,
                'success': info['success'],
                'roll_data': roll_angles,
                'pitch_data': pitch_angles,
                'throttle_data': throttle_data,
                'steering_data': steering_data,
                'vehicle_states': vehicle_states
            })
            
            env.close()
            
        # Calculate statistics for this world
        success_count = sum(1 for r in world_results if r['success'])
        if success_count > 0:
            successful_times = [r['time_to_goal'] for r in world_results if r['success']]
            mean_time = np.mean(successful_times)
            var_time = np.var(successful_times)
        else:
            mean_time = 0
            var_time = 0
            
        all_roll_angles = [angle for r in world_results for angle in r['roll_data']]
        all_pitch_angles = [angle for r in world_results for angle in r['pitch_data']]
        all_throttle = [t for r in world_results for t in r['throttle_data']]
        all_steering = [s for r in world_results for s in r['steering_data']]
        
        world_result = {
            'world_id': world_id,
            'success_rate': success_count / 10,
            'mean_traversal_time': mean_time,
            'variance_traversal_time': var_time,
            'mean_roll': np.mean(all_roll_angles) if all_roll_angles else None,
            'variance_roll': np.var(all_roll_angles) if all_roll_angles else None,
            'mean_pitch': np.mean(all_pitch_angles) if all_pitch_angles else None,
            'variance_pitch': np.var(all_pitch_angles) if all_pitch_angles else None,
            'mean_throttle': np.mean(all_throttle) if all_throttle else None,
            'variance_throttle': np.var(all_throttle) if all_throttle else None,
            'mean_steering': np.mean(all_steering) if all_steering else None,
            'variance_steering': np.var(all_steering) if all_steering else None
        }
        
        # Save world results to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(world_result)
        
        # Update JSON file
        with open(json_path, 'r') as jsonfile:
            results = json.load(jsonfile)
        results.append(world_result)
        with open(json_path, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        
        # Save detailed world results
        world_results_path = os.path.join(res_dir, f'world{world_id}_data.json')
        with open(world_results_path, 'w') as world_jsonfile:
            json.dump(world_results, world_jsonfile, indent=4)
            
        print(f"Results for World {world_id} saved to CSV and JSON.")
    
    print("Evaluation complete!")