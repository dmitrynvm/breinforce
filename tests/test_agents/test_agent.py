import random
from breinforce import agents


def test_agent():
    agent = agents.KuhnAgent(0)
    obs = {
        "player": 0,
        "active": [True, True],
        "button": 1,
        "call": 0,
        "board_cards": [],
        "hole_cards": ["Q♠"],
        "max_raise": 1,
        "min_raise": 1,
        "pot": 2,
        "stacks": [9, 9],
        "street_commits": [0, 0],
    }
    assert agent.act(obs) == 0
    obs["hole_cards"] = ["K♠"]
    assert agent.act(obs) == 0
    obs["hole_cards"] = ["A♠"]
    assert agent.act(obs) == 0
    obs["pot"] = 4
    obs["hole_cards"] = ["Q♠"]
    assert agent.act(obs) == 0
    obs["hole_cards"] = ["K♠"]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1
    obs["hole_cards"] = ["A♠"]
    assert agent.act(obs) == 1
    obs["player"] = ["1"]
    obs["pot"] = 2
    obs["hole_cards"] = ["Q♠"]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1
    obs["hole_cards"] = ["K♠"]
    assert agent.act(obs) == 0
    obs["hole_cards"] = ["A♠"]
    assert agent.act(obs) == 1
    obs["pot"] = 3
    obs["hole_cards"] = ["Q♠"]
    assert agent.act(obs) == 0
    obs["hole_cards"] = ["K♠"]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1
    obs["hole_cards"] = ["A♠"]
    assert agent.act(obs) == 1
