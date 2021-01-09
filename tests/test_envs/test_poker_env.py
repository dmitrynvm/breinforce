import gym
import pytest
from breinforce import agents, envs, errors
from breinforce.games.bropoker import Dealer


def test_poker_env():
    env = gym.make('NolimitHoldemTwoPlayer-v0')
    dealer = Dealer(**envs.configs.NOLIMIT_HOLDEM_TWO_PLAYER)
    env_obs = env.reset()
    dealer_obs = dealer.reset()
    assert list(env_obs.keys()) == list(dealer_obs.keys())
    assert all(env_obs['stacks'] == dealer_obs['stacks'])
    bet = 10
    env_obs, *_ = env.step(bet)
    dealer_obs, *_ = dealer.step(bet)
    assert env_obs['pot'] == dealer_obs['pot']
    assert env.close() is None


def test_register():
    env = gym.make('KuhnTwoPlayer-v0')
    with pytest.raises(errors.NoRegisteredAgentsError):
        env.act({})
    env.register_agents(
        [agents.KuhnAgent(0), agents.KuhnAgent(0)]
    )
    with pytest.raises(errors.EnvironmentResetError):
        env.act({})
    env.register_agents(
        {0: agents.KuhnAgent(0), 1: agents.KuhnAgent(0)}
    )
    obs = env.reset()
    bet = env.act(obs)
    _ = env.step(bet)


def test_errors():
    env = gym.make('NolimitHoldemTwoPlayer-v0')
    with pytest.raises(errors.InvalidAgentConfigurationError):
        env.register_agents(None)
    with pytest.raises(errors.InvalidAgentConfigurationError):
        env.register_agents([None, None])
    with pytest.raises(errors.InvalidAgentConfigurationError):
        env.register_agents([None])
    with pytest.raises(errors.InvalidAgentConfigurationError):
        env.register_agents(
            {4: agents.KuhnAgent(0), 5: agents.KuhnAgent(0)}
        )
