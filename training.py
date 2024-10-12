from stable_baselines3 import PPO
import gymnasium as gym
import customEnv

env = gym.make("customEnv/GridWorld-v0")

from stable_baselines3 import SAC

model = SAC('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model
model.save("dqn_custom_env")


