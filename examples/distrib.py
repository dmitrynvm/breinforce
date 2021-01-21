import gym

import ray
from ray import tune
from ray.rllib.agents import ppo
from breinforce import agents, envs
from ray.tune.registry import register_env


config = {
    "n_players": 6,
    "n_streets": 4,
    "n_suits": 4,
    "n_ranks": 13,
    "n_hole_cards": 2,
    "n_cards_for_hand": 5,
    "rake": 0.05,
    "raise_sizes": [float("inf"), float("inf"), float("inf"), float("inf")],
    "ns_board_cards": [0, 3, 1, 1],
    "blinds": [1, 2, 4, 0, 0, 0],
    "antes": [1, 1, 1, 1, 1, 1],
    "stacks": [200, 200, 200, 200, 200, 200],
    "splits": [0.3, 0.5, 0.75, 1, 2]
}


ray.init()
def env_creator(env_config):
    return BropokerEnv(config)

register_env("bropoker", env_creator)

tune.run(
    "DQN",
    stop={
        "timesteps_total": 100,
    },
    config={
        "env": "bropoker",
        "num_workers": 7,
        "env_config": {
            "unity_worker_id": 52
        },
        "train_batch_size": 500,
    },
    checkpoint_at_end=True,
)
