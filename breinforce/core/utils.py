import json
import gym
import os
import uuid
import random
import string


def configure(config_path):
    ''' Merges the local initialization envs to the global OpenAI Gym list.
    '''
    with open(config_path, 'r') as f:
        game_configs = json.loads(f.read())
        for game in game_configs:
            for name, config in game_configs[game].items():
                game_configs[game][name] = parse(config)
                game_config = game_configs[game]
                env_entry_point = 'breinforce.envs:BropokerEnv'
                env_ids = [env.id for env in gym.envs.registry.all()]
                for env_id, env_config in game_config.items():
                    if env_id not in env_ids:
                        gym.envs.registration.register(
                            id=env_id,
                            entry_point=env_entry_point,
                            kwargs={'config': env_config}
                        )


def flatten(legal_actions):
    out = {}
    for k, v in legal_actions.items():
        if isinstance(v, dict):
            for ik, iv in v.items():
                out[f'raise_{ik}'] = iv
        else:
            out[k] = v
    return out


def guid(size, mode='int'):
    """
    Generates unique object identifier

    Args:
        mode (str): mode of generating identifier

    Returns:
        str: generated identifier
    """
    out = ''
    if mode == 'int':
        out = str(uuid.uuid4().int)[:size]
    else:
        out = ''.join(random.choice(string.ascii_lowercase) for _ in range(size))
    return out


def parse(config):
    for key, vals in config.items():
        if type(vals) == str:
            if vals == 'inf':
                config[key] = float('inf')
        if type(vals) == list:
            for i, val in enumerate(vals):
                if type(val) == str:
                    if val == 'inf':
                        vals[i] = float('inf')
    return config

