import gymnasium as gym
from verti_bench.envs.wheeled.off_road_VertiBench import off_road_art
from gymnasium.utils.env_checker import check_env

render = True
if __name__ == '__main__':
    env = off_road_art()    
    obs, _ = env.reset()
    if render:
        env.render('follow')

    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        '''
        Steering: -1 is right, 1 is left
        '''
        obs, reward, terminated, truncated, info = env.step([0.0, 1.0])
        print(obs, reward)
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break
