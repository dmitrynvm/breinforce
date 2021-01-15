import gym
import pytest
from breinforce import agents, envs, exceptions


def test_exceptions():
    env = gym.make("NolimitHoldemTwoPlayer-v0")
    with pytest.raises(exceptions.InvalidAgentConfigurationError):
        env.register_agents(None)
    with pytest.raises(exceptions.InvalidAgentConfigurationError):
        env.register_agents([None, None])
    with pytest.raises(exceptions.InvalidAgentConfigurationError):
        env.register_agents([None])
    with pytest.raises(exceptions.InvalidAgentConfigurationError):
        env.register_agents(
            {4: agents.KuhnAgent(0), 5: agents.KuhnAgent(0)}
        )
