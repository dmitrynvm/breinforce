import gym
import json
from breinforce import agents, envs, views, utils

utils.configure()
env = gym.make("CustomSixPlayer-v0")
hh = views.HandsView(env)

probs = [
    0.1,  # fold
    0.3,  # call
    0.2,  # half_pot_raise
    0.2,  # one_pot_raise
    0.1,  # two_pot_rais
    0.0   # all_in_raise
]

agents = [agents.RandomAgent(probs)] * 6
env.register(agents)
obs = env.reset()


while True:
    action = env.act(obs)
    obs, rewards, done, info = env.step(action)
    if all(done):
        break

print(hh.render())
