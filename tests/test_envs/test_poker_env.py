import gym
import pytest
from breinforce import agents, envs, errors


def test_errors():
    env = gym.make("NolimitHoldemTwoPlayer-v0")
