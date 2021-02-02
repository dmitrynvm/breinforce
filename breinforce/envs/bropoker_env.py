'''Bropoker environment class for running poker games'''
from datetime import datetime
import gym
import numpy as np
import random
import string
import uuid
from addict import Dict
from breinforce import agents, games, views
from breinforce.core.types import Action, Episode


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

def create_deck(n_suits, n_ranks):
    '''A deck contains at most 52 cards, 13 ranks 4 suits. Any 'subdeck'
    of the standard 52 card deck is valid, i.e. the number of suits
    must be between 1 and 4 and number of ranks between 1 and 13. A
    deck can be tricked to ensure a certain order of cards.

    Parameters
    ----------
    n_suits : int
        number of suits to use in deck
    n_ranks : int
        number of ranks to use in deck
    '''
    out = []
    ranks = games.bropoker.Card.STR_RANKS[-n_ranks:]
    suits = list(games.bropoker.Card.SUITS_TO_INTS.keys())[:n_suits]
    for rank in ranks:
        for suit in suits:
            out.append(games.bropoker.Card(rank + suit))
    return out


def shuffle(cards):
    '''Shuffles the deck. If a tricking order is given, the desired
    cards are placed on the top of the deck after shuffling.

    Returns
    -------
    Deck
        self
    '''
    return random.sample(cards, len(cards))


def deal(cards, n = 1):
    '''Draws cards from the top of the deck. If the number of cards
    to draw exceeds the number of cards in the deck, all cards
    left in the deck are returned.

    Parameters
    ----------
    n : int, optional
        number of cards to draw, by default 1

    Returns
    -------
    List[Card]
        cards drawn from the deck
    '''
    out = []
    for _ in range(n):
        if cards:
            out.append(cards.pop(0))
        else:
            break
    return out


def get_call(state):
    out = 0
    if not all(get_done(state)):
        out = state.commits.max() - state.commits[state.player]
        out = min(out, state.stacks[state.player])
    return out


def get_min_raise(state):
    out = 0
    if not all(get_done(state)):
        out = max(state.straddle, get_call(state) + state.largest)
        out = min(out, state.stacks[state.player])
    return out


def get_max_raise(state):
    out = 0
    if not all(get_done(state)):
        out = min(state.stacks[state.player], state.raise_sizes[state.street])
    return out


def get_done(state):
    if state.street >= state.n_streets or sum(state.alive) <= 1:
        return np.full(state.n_players, 1)
    return np.logical_not(state.alive)


def get_agreed(state):
    max_commit = state.commits.max()
    acted = state.acted == 1
    empty = state.stacks == 0
    pushed = state.commits == max_commit
    folded = np.logical_not(state.alive)
    return acted + empty + pushed + folded


def get_payouts(state):
    # players that have folded lose their actions
    payouts = -1 * state.contribs * np.logical_not(state.alive)
    if sum(state.alive) == 1:
        payouts += state.alive * (state.pot - state.contribs)
    # if last street played and still multiple players alive
    elif state.street >= state.n_streets:
        payouts = evaluate(state)
        payouts -= state.contribs
    if any(payouts > 0):
        state.stacks += payouts + state.contribs
    return payouts

def get_allin(state):
    return state.alive * (state.stacks == 0)

def get_valid_actions(state):
    call = get_call(state)
    min_raise = get_min_raise(state)
    max_raise = get_max_raise(state)
    #print(call, min_raise, max_raise)
    out = {}
    out['fold'] = 0
    out['call'] = call
    out['raise'] = {
        'min': min_raise,
        'max': max_raise
    }
    out['allin'] = max_raise
    return Dict(out)


def get_observation(state):
    community_cards = [str(c) for c in state.community_cards]
    hole_cards = [[str(c) for c in cs] for cs in state.hole_cards]
    obs = {
        'street': state.street,
        'button': state.button,
        'player': state.player,
        'pot': state.pot,
        'community_cards': community_cards,
        'hole_cards': hole_cards[state.player],
        'alive': state.alive.tolist(),
        'stacks': state.stacks.tolist(),
        'commits': state.commits.tolist(),
        'valid_actions': get_valid_actions(state)
    }
    return obs


def evaluate(state) -> np.ndarray:
    judge = games.bropoker.Judge(state.n_suits, state.n_ranks, state.n_cards_for_hand)
    # grab array of hand strength and pot contribs
    worst_hand = judge.hashmap.max_rank + 1
    hand_list = []
    payouts = np.zeros(state.n_players, dtype=int)
    for player in range(state.n_players):
        # if not alive hand strength set
        # to 1 worse than worst possible rank
        hand_strength = worst_hand
        if state.alive[player]:
            hand_strength = judge.evaluate(
                state.hole_cards[player], state.community_cards
            )
        hand_list.append([player, hand_strength, state.contribs[player]])
    hands = np.array(hand_list)
    # sort hands by hand strength and pot contribs
    hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
    pot = state.pot
    remainder = 0
    # iterate over hand strength and
    # pot contribs from smallest to largest
    for idx, (_, strength, contribs) in enumerate(hands):
        eligible = hands[:, 0][hands[:, 1] == strength].astype(int)
        # cut can only be as large as lowest player commit amount
        cut = np.clip(hands[:, 2], None, contribs)
        split_pot = sum(cut)
        split = split_pot // len(eligible)
        remain = split_pot % len(eligible)
        payouts[eligible] += split
        remainder += remain
        # remove chips from players and pot
        hands[:, 2] -= cut
        pot -= split_pot
        # remove player from move split pot
        hands[idx, 1] = worst_hand
        if pot == 0:
            break
    # give worst position player remainder chips
    if remainder:
        # worst player is first player after button involved in pot
        involved_players = np.nonzero(payouts)[0]
        button_shift = (involved_players <= state.button) * state.n_players
        button_shifted_players = involved_players + button_shift
        worst_idx = np.argmin(button_shifted_players)
        worst_pos = involved_players[worst_idx]
        payouts[worst_pos] += remainder
    return payouts


def move_player(state):
    for idx in range(1, state.n_players + 1):
        player = (state.player + idx) % state.n_players
        if state.alive[player]:
            break
        else:
            state.acted[player] = True
    state.player = player


def move_street(state):
    if all(get_agreed(state)):
        state.player = state.button
        move_player(state)
        # if at most 1 player alive and not all in turn up all
        # board cards and evaluate hand
        while True:
            state.street += 1
            allin = state.alive * (state.stacks == 0)
            all_allin = sum(state.alive) - sum(allin) <= 1
            if state.street >= state.n_streets:
                break
            state.community_cards += deal(
                state.deck,
                state.ns_community_cards[state.street]
            )
            if not all_allin:
                break
        state.commits.fill(0)
        state.acted = np.logical_not(state.alive).astype(int)


def perform_action(state, action):
    action_value = max(0, min(state.stacks[state.player], action.value))
    state.pot += action_value
    state.contribs[state.player] += action_value
    state.commits[state.player] += action_value
    state.stacks[state.player] -= action_value
    state.acted[state.player] = 1


def perform_antes(state):
    actions = state.antes
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.contribs += actions
    state.stacks -= actions


def perform_blinds(state):
    actions = state.blinds
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.commits += actions
    state.contribs += actions
    state.stacks -= actions


def update_largest(state, action):
    valid_actions = get_valid_actions(state)
    if action.value and (action.value - valid_actions['call']) >= state.largest:
        state.largest = action.value - valid_actions['call']


def update_folded(state, action):
    valid_actions = get_valid_actions(state)
    if 'call' in valid_actions:
        if valid_actions.call and ((action.value < valid_actions.call) or action.value < 0):
            state.alive[state.player] = 0
            state.folded[state.player] = state.street
    else:
        state.alive[state.player] = 0
        state.folded[state.player] = state.street


def update_payouts(state, payouts):
    state.payouts = payouts


def update_valid_actions(state):
    state.valid_actions = get_valid_actions(state)


def create_state(config):
    n_players = config['n_players']
    n_streets = config['n_streets']
    n_ranks = config['n_ranks']
    n_suits = config['n_suits']
    n_hole_cards = config['n_hole_cards']
    n_community_cards = config['ns_community_cards'][0]
    deck = shuffle(create_deck(n_suits, n_ranks))
    state = Dict({
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
        'game': guid(9, 'int'),
        'table': guid(5, 'str'),
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
        'payouts': [0 for _ in range(n_players)],
        'community_cards': deal(deck, n_community_cards),
        'hole_cards': [deal(deck, n_hole_cards) for _ in range(n_players)],
        'alive': np.ones(n_players, dtype=np.uint8),
        'contribs': np.zeros(n_players, dtype=np.int32),
        'acted': np.zeros(n_players, dtype=np.uint8),
        'commits': np.zeros(n_players, dtype=np.int32),
        'folded': np.array([n_streets for i in range(n_players)]),
        'deck': deck,
        'valid_actions': None
    })
    if n_players > 2:
        move_player(state)
    #perform_antes(state)
    perform_blinds(state)
    move_player(state)
    move_player(state)
    move_player(state)
    return state


def move_step(state, action):
    va = get_valid_actions(state)
    update_largest(state, action)
    update_folded(state, action)
    update_valid_actions(state)
    perform_action(state, action)
    move_player(state)
    move_street(state)

    observation = get_observation(state)
    payouts = get_payouts(state)
    update_payouts(state, payouts)
    done = get_done(state)
    info = None
    observation['hole_cards'] = observation['hole_cards']
    return observation, payouts, done, info


class BropokerEnv(gym.Env):

    def __init__(self, config):
        self.config = config
        self.state = create_state(config)
        self.history = []
        self.agents = None

    def register(self, players):
        self.players = players

    def predict(self, obs):
        return self.players[self.state.player].predict(obs)

    def reset(self):
        self.state = create_state(self.config)
        self.history = []
        episode = Episode(self.state.deepcopy(), None, None, None)
        self.history += [episode]
        return get_observation(self.state)

    def step(self, action):
        player = self.state.player
        out = move_step(self.state, action)
        self.history += [Episode(self.state.deepcopy(), player, action, None)]
        return out

    def render(self, mode='pokerstars'):
        out = None
        if mode == 'pokerstars':
            out = views.pokerstars.render(self)
        elif mode == 'jsonify':
            out = views.jsonify.render(self.history)
        return out
