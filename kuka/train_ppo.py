from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import KukaReachVisualEnv

env = make_vec_env(KukaReachVisualEnv, 
                   n_envs=1)

model = PPO("CnnPolicy", 
            env, 
            verbose=1
        )

model.learn(total_timesteps=25000, progress_bar=True)

model.save("kuka_ppo")

