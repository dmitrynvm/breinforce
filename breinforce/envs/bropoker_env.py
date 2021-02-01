'''Bropoker environment class for running poker games'''
from datetime import datetime
import gym
import numpy as np
import random
import uuid
from addict import Dict
from breinforce import agents, games, views
from breinforce.core.types import Action, Episode


def guid():
    return str(uuid.uuid4().int)[:11]


def deck(n_suits, n_ranks):
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


def get_call(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = state.commits.max() - state.commits[state.player]
        out = min(out, state.stacks[state.player])
    return out


def get_min_raise(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = max(state.straddle, get_call(state) + state.largest)
        out = min(out, state.stacks[state.player])
    return out


def get_max_raise(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = min(state.stacks[state.player], state.raise_sizes[state.street])
    return out


def get_done(state):
    if state.street >= state.n_streets or sum(state.alive) <= 1:
        return np.full(state.n_players, 1)
    return np.logical_not(state.alive)


def get_agreed(state) -> bool:
    max_commit = state.commits.max()
    acted = state.acted == 1
    empty = state.stacks == 0
    committed = state.commits == max_commit
    alived = np.logical_not(state.alive)
    return acted + empty + committed + alived


def get_payouts(state) -> np.ndarray:
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


def get_valid_actions(state):
    call = get_call(state)
    min_raise = get_min_raise(state)
    max_raise = get_max_raise(state)
    out = {}
    out['fold'] = -1
    if not call:
        out['check'] = 0
    out['call'] = call
    out['raise'] = {'min': min_raise, 'max': max_raise}
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
        'call': get_call(state),
        'max_raise': get_max_raise(state),
        'min_raise': get_min_raise(state),
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
        # remove player from next split pot
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


def next_player(state):
    for idx in range(1, state.n_players + 1):
        player = (state.player + idx) % state.n_players
        if state.alive[player]:
            break
        else:
            state.acted[player] = True
    state.player = player


def perform_action_(state, action):
    action = min(state.stacks[state.player], action)
    state.pot += action
    state.contribs[state.player] += action
    state.commits[state.player] += action
    state.stacks[state.player] -= action
    state.acted[state.player] = True


def perform_antes_(state):
    actions = state.antes
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.contribs += actions
    state.stacks -= actions


def perform_blinds_(state):
    actions = state.blinds
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.commits += actions
    state.contribs += actions
    state.stacks -= actions


def create_state(config):
    n_players = config['n_players']
    n_streets = config['n_streets']
    n_ranks = config['n_ranks']
    n_suits = config['n_suits']
    n_hole_cards = config['n_hole_cards']
    n_board_cards = config['ns_board_cards'][0]
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
        'ns_board_cards': config["ns_board_cards"],
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
        'board_cards': deal(deck, n_board_cards),
        'hole_cards': [deal(deck, n_hole_cards) for _ in range(n_players)],
        'alive': np.ones(n_players, dtype=np.uint8),
        'contribs': np.zeros(n_players, dtype=np.int32),
        'acted': np.zeros(n_players, dtype=np.uint8),
        'commits': np.zeros(n_players, dtype=np.int32),
        'folded': [n_streets for i in range(n_players)],
        'deck': deck
    })
    if n_players > 2:
        next_player(state)
    perform_antes_(state)
    perform_blinds_(state)
    next_player(state)
    next_player(state)
    next_player(state)
    return state


def update_largest(state, action):
    va = get_valid_actions(state)
    if action.value and (action.value - va.call) >= state.largest:
        state.largest = action.value - va.call


def update_folded(state, action):
    va = get_valid_actions(state)
    if va.call and ((action.value < va.call) or action.value < 0):
        state.alive[state.player] = 0
        state.folded[state.player] = state.street


def next_street(state):
    if all(get_agreed(state)):
        state.player = state.button
        next_player(state)
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


class BropokerEnv(gym.Env):
    def __init__(self, config) -> None:

        # envs
        self.n_players = config['n_players']
        self.n_streets = config['n_streets']
        self.n_suits = config['n_suits']
        self.n_ranks = config['n_ranks']
        self.n_hole_cards = config['n_hole_cards']
        self.n_cards_for_hand = config['n_cards_for_hand']
        self.rake = config['rake']
        self.raise_sizes = config['raise_sizes']
        self.ns_community_cards = config['ns_community_cards']
        self.blinds = np.array(config['blinds'], dtype=int)
        self.antes = np.array(config['antes'], dtype=int)
        self.splits = config['splits']
        self.start_stacks = np.array(config['stacks'], dtype=int)
        self.stacks = np.array(config['stacks'], dtype=int)

        # agnt
        self.small_blind = config['blinds'][0]
        self.big_blind = config['blinds'][1] if self.n_players > 2 else None
        self.straddle = config['blinds'][2] if self.n_players > 3 else None
        self.game = guid()
        self.date = datetime.now()
        self.table = 'Table_1'
        self.player_names = ['agent_' + str(i+1) for i in range(self.n_players)]
        self.hole_cards = []
        self.community_cards = []
        self.payouts = None

        # dealer
        self.street = 0
        self.button = 0
        self.player = -1
        self.largest = 0
        self.pot = 0
        self.alive = np.zeros(self.n_players, dtype=np.uint8)
        self.contribs = np.zeros(self.n_players, dtype=np.int32)
        self.payouts = []

        self.acted = np.zeros(self.n_players, dtype=np.uint8)
        self.commits = np.zeros(self.n_players, dtype=np.int32)
        self.folded = [self.n_streets for i in range(self.n_players)]
        self.history = []

        self.deck = shuffle(deck(self.n_suits, self.n_ranks))
        self.players = None

        max_action = sum(self.stacks)
        self.action_space = gym.spaces.Discrete(max_action)
        n_community_cards = sum(self.ns_community_cards)
        card_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n_ranks), gym.spaces.Discrete(self.n_suits))
        )
        hole_card_space = gym.spaces.Tuple((card_space,) * self.n_hole_cards)
        '''
            'alive': gym.spaces.MultiBinary(self.n_players),
            'action': gym.spaces.Discrete(self.n_players),
            'button': gym.spaces.Discrete(self.n_players),
            'call': gym.spaces.Discrete(max_action),
            'max_raise': gym.spaces.Discrete(max_action),
            'min_raise': gym.spaces.Discrete(max_action),
            'street_commits': gym.spaces.Tuple(
                (gym.spaces.Discrete(max_action),) * self.n_players
            )
            'stacks': gym.spaces.Tuple((gym.spaces.Discrete(max_action),) * self.n_players),
        '''

        self.observation_space = gym.spaces.Dict(
            {
                'community_cards': gym.spaces.Tuple((card_space,) * n_community_cards),
                'hole_cards': gym.spaces.Tuple((hole_card_space,) * self.n_players),
                'pot': gym.spaces.Discrete(max_action)
            }
        )

    def register(self, players):
        self.players = players

    def predict(self, obs):
        return self.players[self.player].predict(obs)

    def reset(self):
        '''Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        '''
        self.alive.fill(1)
        self.stacks = self.start_stacks.copy()
        self.button = 0
        self.deck = shuffle(deck(self.n_suits, self.n_ranks))
        self.community_cards = deal(self.deck, self.ns_community_cards[0])
        self.history = []
        self.hole_cards = [
            deal(self.deck, self.n_hole_cards) for _
            in range(self.n_players)
        ]
        self.player = self.button
        self.street = 0
        self.pot = 0
        self.largest = self.blinds[2]
        self.contribs.fill(0)
        self.commits.fill(0)
        self.acted.fill(0)
        if self.n_players > 2:
            next_player(self)
        # perform_antes_(self)
        perform_blinds_(self)
        next_player(self)
        next_player(self)
        next_player(self)
        return get_observation(self)

    def step(self, action):
        call = get_call(self)
        info = {
            'call': call,
            'min_raise': get_min_raise(self),
            'max_raise': get_max_raise(self)
        }

        update_largest(self, action)
        update_folded(self, action)

        self.history.append(Episode(self.state, self.player, action, info))
        perform_action_(self, action.value)
        next_player(self)

        # if all agreed go to next street
        next_street(self)

        observation = get_observation(self)
        payouts = get_payouts(self)
        self.payouts = payouts
        done = get_done(self)
        info = None
        observation['hole_cards'] = observation['hole_cards']
        return Episode(observation, payouts, done, info)

    def render(self, mode='pokerstars'):
        out = None
        if mode == 'pokerstars':
            out = views.pokerstars.render(self)
        return out

    @property
    def state(self):
        out = {
            'table': self.table,
            'street': self.street,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'straddle': self.straddle,
            'game': self.game,
            'date': self.date,
            'player_names': self.player_names,
            'hole_cards': self.hole_cards,
            'start_stacks': self.start_stacks,
            'player': self.player,
            'community_cards': self.community_cards,
            'button': self.button,
            'pot': self.pot,
            'payouts': get_payouts(self),
            'n_players': self.n_players,
            'n_hole_cards': self.n_hole_cards,
            'ns_community_cards': sum(self.ns_community_cards),
            'rake': self.rake,
            'antes': self.antes,
            'valid_action': get_valid_actions(self),
            'min_raise': get_min_raise(self),
            'max_raise': get_max_raise(self),
            'acted': self.acted.copy(),
            'alive': self.alive.copy(),
            'stacks': self.stacks.copy(),
            'commits': self.commits.copy(),
            'alives': self.alive.copy(),
            'folded': self.folded.copy()
        }
        return out
