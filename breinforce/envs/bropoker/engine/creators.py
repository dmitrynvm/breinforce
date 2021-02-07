'''Bropoker environment class for running poker games'''
from datetime import datetime
import numpy as np
import random
from addict import Dict
from breinforce import agents, core, views
from breinforce.envs.bropoker.types import Deck


def reset():
    return {"type": "RESET"}


def step(action):
    return {
        "type": "STEP",
        'action': action
    }


def init_state(config):
    n_players = config['n_players']
    n_streets = config['n_streets']
    n_ranks = config['n_ranks']
    n_suits = config['n_suits']
    n_hole_cards = config['n_hole_cards']
    n_community_cards = config['ns_community_cards'][0]
    deck = Deck(n_suits, n_ranks)
    out = Dict({
        'n_players': config["n_players"],
        'n_streets': config["n_streets"],
        'n_suits': config["n_suits"],
        'n_ranks': config["n_ranks"],
        'n_hole_cards': config["n_hole_cards"],
        'n_cards_for_hand': config["n_cards_for_hand"],
        'rake': config['rake'],
        'raise_sizes': config['raise_sizes'],
        'ns_community_cards': config["ns_community_cards"],
        'blinds': np.array(config["blinds"], dtype=int),
        'antes': np.array(config["antes"], dtype=int),
        'splits': np.array(config["splits"], dtype=int),
        'stacks': np.array(config["stacks"], dtype=int),
        # meta
        'game': core.utils.guid(9, 'int'),
        'table': core.utils.guid(5, 'str'),
        'date': datetime.now(),
        'player_names': ["agent_" + str(i+1) for i in range(n_players)],
        'small_blind': config['blinds'][0],
        'big_blind': config['blinds'][1] if n_players > 1 else None,
        'straddle': config['blinds'][2] if n_players > 3 else None,
        # dealer
        'street': 0,
        'button': 0,
        'player': 0,
        'largest': 0,
        'pot': 0,
        'rewards': [0 for _ in range(n_players)],
        'community_cards': deck.deal(n_community_cards),
        'hole_cards': [deck.deal(n_hole_cards) for _ in range(n_players)],
        'alive': np.ones(n_players, dtype=np.uint8),
        'contribs': np.zeros(n_players, dtype=np.int32),
        'acted': np.zeros(n_players, dtype=np.uint8),
        'commits': np.zeros(n_players, dtype=np.int32),
        'folded': np.array([n_streets for i in range(n_players)]),
        'deck': deck,
        'valid_actions': None
    })
    return out
