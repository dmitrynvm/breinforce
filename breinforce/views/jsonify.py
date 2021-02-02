import json
from addict import Dict
import numpy as np
#import pprint; pp = pprint.PrettyPrinter(indent=4).pprint

def sub(state):
    out = {}
    keys = ['largest', 'stacks', 'payouts', 'community_cards', 'hole_cards', 'alive', 'contribs', 'acted', 'commits', 'folded', 'valid_actions']

    for key in keys:
        val = state[key]
        if isinstance(val, np.ndarray):
            out[key] = str(val.tolist())
        elif isinstance(val, list):
            out[key] = str(val)
        else:
            out[key] = str(val)
        out['hole_cards'] = str(state['hole_cards'][state.player])
    return out


def render(history):
    out = ''
    for episode in history:
        player = episode.player
        street = episode.state.street
        out += f'player: {player}, street: {street}, '
        out += str(episode.action) + '\n'
        out += json.dumps(sub(episode.state), sort_keys=True, indent=4) + '\n'
    return out
