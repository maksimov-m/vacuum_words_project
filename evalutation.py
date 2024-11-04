import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from customEnv.envs.grid_world import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.preprocessing import is_image_space
# Load the trained model

# Create the environment
env = gym.make("customEnv/GridWorld-v0", render_mode="human")

# Number of episodes to evaluate
model = SAC.load("sac_custom_env")
# a2c ppo
# Evaluate the model
obs, info = env.reset()
for _ in range(10000):
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs, info = env.reset()