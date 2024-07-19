from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import KukaReachVisualEnv

env = make_vec_env(KukaReachVisualEnv, n_envs=1,env_kwargs={"is_render":True})

model = PPO.load("kuka_ppo")

obs = env.reset()
dones = False
while dones==False:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
