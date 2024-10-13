from stable_baselines3 import PPO
import gymnasium as gym
import customEnv

env = gym.make("customEnv/GridWorld-v0")

from stable_baselines3 import PPO


model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=1000, progress_bar=True)

# Save the model
model.save("sac_custom_env")


