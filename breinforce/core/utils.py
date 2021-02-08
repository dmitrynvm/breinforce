import json
import gym
import os
import uuid
import random
import string
from breinforce.envs import configs


def configure():
    ''' Merges the local initialization envs to the global OpenAI Gym list.
    '''
    game_configs = configs.__dict__
    for game_name, env_configs in game_configs.items():
        if game_name.isupper():
            env_entry_point = 'breinforce.envs:BropokerEnv'
            env_names = [env.id for env in gym.envs.registry.all()]
            for env_name, env_config in env_configs.items():
                if env_name not in env_names:
                    print(env_name)
                    gym.envs.registration.register(
                        id=env_name,
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
