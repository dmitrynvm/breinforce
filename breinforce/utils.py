import json
import gym
import os
from breinforce.config.application import CONFIG_DIR


def parse_config(config: dict) -> dict:
    for key, vals in config.items():
        if type(vals) == str:
            if vals == "inf":
                config[key] = float("inf")
        if type(vals) == list:
            for i, val in enumerate(vals):
                if type(val) == str:
                    if val == "inf":
                        vals[i] = float("inf")
    return config


def configure():
    """ Merges the local initialization envs to the global OpenAI Gym list.
    """
    config_path = os.path.join(CONFIG_DIR, "envs.json")
    with open(config_path, "r") as f:
        game_configs = json.loads(f.read())
        for game in game_configs:
            for name, config in game_configs[game].items():
                game_configs[game][name] = parse_config(config)
                game_config = game_configs[game]
                env_entry_point = "breinforce.envs:Bropoker"
                env_ids = [env.id for env in gym.envs.registry.all()]
                for env_id, env_config in game_config.items():
                    if env_id not in env_ids:
                        gym.envs.registration.register(
                            id=env_id,
                            entry_point=env_entry_point,
                            kwargs=env_config
                        )
