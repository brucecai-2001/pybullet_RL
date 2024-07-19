import gym
from stable_baselines3 import PPO

env = gym.make("PandaReach-v2")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=25000, progress_bar=True)
model.save("panda_ppo")