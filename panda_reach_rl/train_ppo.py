import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO

env = gym.make(
    "PandaReach-v3",
    render_mode="human",
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

# Set up PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    learning_rate=0.001,
)

# Train agent
model.learn(total_timesteps=200000, progress_bar=True)

# Save trained model
model.save("PandaReach_PPO")