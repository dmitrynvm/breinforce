"""
Rule-based agents Kuhn poker game example
"""

import gym
from breinforce.agents import KuhnAgent

env = gym.make('KuhnTwoPlayer-v0')

env.register_agents([KuhnAgent(0.3)] * 2)

obs = env.reset()

while True:
    bet = env.act(obs)
    obs, rewards, done, info = env.step(bet)

    if all(done):
        break

print(rewards)
