from stable_baselines3 import PPO
import gymnasium as gym
import customEnv

env = gym.make("customEnv/GridWorld-v0")

from stable_baselines3 import SAC


model = SAC('MlpPolicy', env, verbose=1, batch_size=1024, gamma=0.9, buffer_size=10000000)

model.learn(total_timesteps=30000, progress_bar=False)

# Save the model
model.save("sac_custom_env_new")


