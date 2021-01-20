# import gym
# from stable_baselines3 import DQN
# from stable_baselines3.dqn import MlpPolicy
# from breinforce import agents, envs, views, utils
# utils.configure()

# env = gym.make("CustomSixPlayer-v0")

# model = DQN(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_pendulum")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = gym.make('CartPole-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_pendulum")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()