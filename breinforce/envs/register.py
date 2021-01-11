import gym
from typing import Dict


def register(configs: Dict) -> None:
    '''Registers dict of breinforce configs as gym environments

    Parameters
    ----------
    configs : Dict
        dictionary of breinforce configs, keys must environment ids and
        values valid breinforce configs, example:
            configs = {
                'NolimitHoldemTwoPlayer-v0': {
                    'n_players': 2,
                    'n_streets': 4,
                    'blinds': [1, 2],
                    'antes': 0,
                    'raise_sizes': float('inf'),
                    'n_raises': float('inf'),
                    'n_suits': 4,
                    'n_ranks': 13,
                    'n_hole_cards': 2,
                    'n_community_cards': [0, 3, 1, 1],
                    'n_cards_for_hand': 5,
                    'start_stack': 200
                }
            }
    '''
    env_entry_point = 'breinforce.envs:Bropoker'
    env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]
    for env_id, config in configs.items():
        if env_id not in env_ids:
            gym.envs.registration.register(
                id=env_id, entry_point=env_entry_point, kwargs={**config}
            )
        
