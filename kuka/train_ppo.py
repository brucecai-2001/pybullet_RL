from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import KukaReachVisualEnv

kuka_env = KukaReachVisualEnv()
# check_env(kuka_env, warn=True)

model = PPO("CnnPolicy", kuka_env, policy_kwargs=dict(normalize_images=False), verbose=1)
model.learn(total_timesteps=25000)
model.save("kuka_ppo")

del model # remove to demonstrate saving and loading

model = PPO.load("kuka_ppo")

obs, _ = kuka_env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = kuka_env.step(action)
    kuka_env.render("human")