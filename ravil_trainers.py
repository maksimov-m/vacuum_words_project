from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
import gymnasium as gym
import customEnv

env = gym.make("customEnv/GridWorld-v0")

from stable_baselines3 import TD3

model_ppo = PPO('MlpPolicy', env, verbose=1)
model_td3 = TD3('MlpPolicy', env, verbose=1)
model_ddpg = DDPG('MlpPolicy', env, verbose=1)
model_a2c = A2C('MlpPolicy', env, verbose=1)
model_sac = SAC('MlpPolicy', env, verbose=1)

arr_model_path = ["ppo", 'td3', 'ddpg', 'a2c', 'sac']
arr_model = [model_ppo, model_td3, model_ddpg, model_a2c, model_sac]

for model, name in zip(arr_model, arr_model_path):
    try:
        model.learn(total_timesteps=200000, progress_bar=False)
        model.save(f"{name}_custom_env")
    except Exception as ex:
        print(name, "error!!")
        print(ex)

