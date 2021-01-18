"""
Rule-based agents Kuhn poker game example
"""

import gym
from breinforce.agents import KuhnAgent

env = gym.make("KuhnTwoPlayer-v0")

env.register([KuhnAgent(0.3)] * 2)

obs = env.reset()

while True:
    action = env.act(obs)
    obs, rewards, done, _ = env.step(action)

    if all(done):
        break

print(rewards)
