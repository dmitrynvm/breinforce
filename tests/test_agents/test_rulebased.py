import random
import gym
from breinforce import agents, core, envs


def test_rulebased():
    core.utils.configure()
    env = gym.make('CustomSixPlayer-v0')

    splits = [1/3, 1/2, 3/4, 1, 2]
    players = [agents.RuleBasedAgent(splits)] * 6
    env.register(players)
    obs = env.reset()

    while True:
        action = env.predict(obs)
        obs, reword, done = env.step(action)
        if all(done):
            break
    assert isinstance(env.render(), str)
