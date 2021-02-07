import json
from addict import Dict
import numpy as np
#import pprint; pp = pprint.PrettyPrinter(indent=4).pprint

def sub(state):
    out = {}
    keys = ['largest', 'pot', 'stacks', 'rewards', 'community_cards', 'hole_cards', 'alive', 'contribs', 'acted', 'commits', 'folded', 'valid_actions']

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


def render(episodes):
    out = ''
    prev_state = None
    for episode in episodes:
        player = episode.player
        if prev_state:
            street = prev_state.street
        else:
            street = 0
        out += f'player: {player}, street: {street}, '
        out += str(episode.action) + '\n'
        out += json.dumps(sub(episode.state), sort_keys=True, indent=4) + '\n'
        prev_state = episode.state
    return out
