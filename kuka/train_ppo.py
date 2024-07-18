from stable_baselines3 import PPO
from env import KukaReachVisualEnv, CustomSkipFrame

# Parallel environments
kuka_env = KukaReachVisualEnv()
kuka_env = CustomSkipFrame(env=kuka_env)

model = PPO("CnnPolicy", kuka_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("kuka_ppo")

del model # remove to demonstrate saving and loading

model = PPO.load("kuka_ppo")

obs = kuka_env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = kuka_env.step(action)
    kuka_env.render("human")