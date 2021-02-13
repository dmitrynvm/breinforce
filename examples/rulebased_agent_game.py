import gym
import json
from breinforce import agents, core, envs, views

core.utils.configure()
env = gym.make('CustomSixPlayer-v0')

splits = [1/3, 1/2, 3/4, 1, 3/2]
players = [agents.RuleBasedAgent(splits)] * 6
env.register(players)
obs = env.reset()


while True:
    action = env.predict(obs)
    obs, rewards, done = env.step(action)
    print(rewards)
    if all(done):
        break

print(env.render())
