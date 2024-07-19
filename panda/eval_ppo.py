import gymnasium as gym
import panda_gym
import torch as th
from stable_baselines3 import PPO
import time

"""
Run a trained agent in the PandaReach-v3 environment.

Author: Simon Bøgh
"""
#############################
# Load environment
env = gym.make(
    "PandaReach-v3",
    render_mode="human",
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

#############################
# Load trained model
model = PPO.load(
    "PandaReach_PPO",
    print_system_info=True,
    device="auto",
)

#############################
# Reset environment and get first observation
obs, info = env.reset()

#############################
# Run trained agent in environment
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()  # Rendering in real-time (1x)
    # Sleep for 0.1 seconds to slow down the rendering
    time.sleep(0.1)

    if dones:
        obs, info = env.reset()