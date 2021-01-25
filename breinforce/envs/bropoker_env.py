"""Bropoker environment class for running poker games"""

from datetime import datetime
import json
import gym
import numpy as np
import uuid
import pprint
from addict import Dict
from collections import namedtuple
from breinforce.agents import BaseAgent
from breinforce.games.bropoker import Deck, Judge
from breinforce.views import AsciiView, HandsView
pp = pprint.PrettyPrinter(indent=4)


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
        out = str(uuid.uuid4().hex)[:size]
    return out


def date():
    """
    Generates unuque object identifier

    Returns:
        str: generated formated date
    """
    return datetime.now().strftime("%d %m %Y %H:%M:%S")


def clean(legal_actions, action) -> int:
    """
    Find closest (type, action) pair to given one.

    Args:
        action (int): betting amount

    Returns:
        tuple: (type, action)
    """
    vals = list(legal_actions.values())
    items = list(legal_actions.items())
    index = np.argmin(np.absolute(np.array(vals) - action))
    return items[index]


def get_call(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = state.commits.max() - state.commits[state.player]
        print(out)
        out = min(out, state.stacks[state.player])
        print(out)
    return out


def get_min_raise(state) -> int:
    out = 0
    if not all(get_done(state)):
        print(state.straddle)
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


def get_agree(state) -> bool:
    max_commit = state.commits.max()
    acted = state.acted == 1
    empty = state.stacks == 0
    committed = state.commits == max_commit
    alived = np.logical_not(state.alive)
    return acted + empty + committed + alived


def get_legal_actions(state):
    call = get_call(state)
    #min_raise = get_min_raise(state)
    max_raise = get_max_raise(state)
    n_splits = len(state.splits)
    raises = [f'raise_{i}' for i in range(n_splits)]
    out = {}
    out['fold'] = -1
    out['check'] = 0
    out['call'] = call
    for i, split in enumerate(state.splits):
        action = call + int(split * state.pot)
        if action < max_raise:
            out[f'raise_{i+1}'] = action
    out['push'] = max_raise
    return out


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


def observe(state):
    out = None
    board_cards = [str(c) for c in state.board_cards]
    hole_cards = [[str(c) for c in cs] for cs in state.hole_cards]
    out: dict = {
        "street": state.street,
        "button": state.button,
        "player": state.player,
        "pot": state.pot,
        "call": get_call(state),
        "min_raise": get_min_raise(state),
        "max_raise": get_max_raise(state),
        "board_cards": list(board_cards),
        "hole_cards": list(hole_cards[state.player]),
        "alive": list(state.alive),
        "stacks": list(state.stacks),
        "commits": list(state.commits),
        "legal_actions": state.legal_actions
    }

    return out



def evaluate(state) -> np.ndarray:
    # grab array of hand strength and pot contribs
    worst_hand = state.judge.hashmap.max_rank + 1
    hand_list = []
    payouts = np.zeros(state.n_players, dtype=int)
    for player in range(state.n_players):
        # if not alive hand strength set
        # to 1 worse than worst possible rank
        hand_strength = worst_hand
        if state.alive[player]:
            hand_strength = state.judge.evaluate(
                state.hole_cards[player], state.board_cards
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


def move_(state):
    for idx in range(1, state.n_players + 1):
        player = (state.player + idx) % state.n_players
        if state.alive[player]:
            state.acted[player] = 1
            break
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


deck = None
judge = None

def create_state(config):
    global deck, judge
    n_players = config['n_players']
    n_streets = config['n_streets']
    n_ranks = config['n_ranks']
    n_suits = config['n_suits']
    n_hole_cards = config['n_hole_cards']
    n_board_cards = config['ns_board_cards'][0]
    n_cards_for_hand = config['n_cards_for_hand']

    deck = Deck(n_suits, n_ranks)
    judge = Judge(n_suits, n_ranks, n_cards_for_hand)

    body = {
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
        'game_name': guid(9, 'int'),
        'table_name': guid(5, 'int'),
        'date': date(),
        'player_names': ["agent_" + str(i+1) for i in range(n_players)],
        # dealer
        'street': 0,
        'button': 0,
        'player': 0,
        'largest': 0,
        'pot': 0,
        'payouts': [0 for _ in range(n_players)],
        'board_cards': deck.draw(n_board_cards),
        'hole_cards': [deck.draw(n_hole_cards) for _ in range(n_players)],
        'alive': np.ones(n_players, dtype=np.uint8),
        'contribs': np.zeros(n_players, dtype=np.int32),
        'acted': np.zeros(n_players, dtype=np.uint8),
        'commits': np.zeros(n_players, dtype=np.int32),
        'folded': [n_streets for i in range(n_players)],
        'legal_actions': Dict()
    }

    return Dict(body)


class BropokerEnv(gym.Env):

    def __init__(self, config) -> None:
        self.config = config
        self.state = create_state(config)
        self.history = []
        self.agents = None

    def register(self, agents) -> None:
        self.agents = agents

    def act(self, obs: dict) -> int:
        return self.agents[obs.player].act(obs)

    def reset(self):
        """Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        self.state = create_state(self.config)
        self.history = []
        if self.state.n_players > 2:
            move_(self.state)
        pp.pprint(self.state)
        min_raise = get_min_raise(self.state)
        '''
        # perform_antes_(self)
        perform_blinds_(self)
        move_(self)
        move_(self)
        move_(self)
        '''

        return #observe(self.state)

    def step(self, action: int):
        legal_actions = get_legal_actions(self)
        type_, action = clean(legal_actions, action)
        call = get_call(self)
        info = {
            "action_type": type_,
            "call": call,
            "min_raise": get_min_raise(self),
            "max_raise": get_max_raise(self),
            "stack": self.stacks[self.player]
        }

        # larg
        if action and (action - call) >= self.largest:
            self.largest = action - call

        # fold
        if call and ((action < call) or action < 0):
            action = 0
            self.alive[self.player] = 0
            self.folded[self.player] = self.street

        self.history.append((self.state, self.player, action, info))
        perform_action_(self, action)
        move_(self)

        # if all agreed go to next street
        if all(get_agree(self)):
            self.player = self.button
            move_(self)
            # if at most 1 player alive and not all in turn up all
            # board cards and evaluate hand
            while True:
                self.street += 1
                allin = self.alive * (self.stacks == 0)
                all_allin = sum(self.alive) - sum(allin) <= 1
                if self.street >= self.n_streets:
                    break
                self.board_cards += self.deck.draw(
                    self.ns_board_cards[self.street]
                )
                if not all_allin:
                    break
            self.commits.fill(0)
            self.acted = np.logical_not(self.alive).astype(np.uint8)

        obs = get_observation(self)
        payouts = get_payouts(self)
        self.payouts = payouts
        done = get_done(self)
        info = None
        if all(done):
            self.player = -1
            obs["player"] = -1
        obs["hole_cards"] = obs["hole_cards"]#[obs["player"]]
        return obs, payouts, done, info

    def render(self, mode="hands"):
        out = None
        hands_view = HandsView(self)
        if mode == "hands":
            out = hands_view.render()
        return out

    @property
    def observation_space(self):
        max_action = sum(self.stacks)
        n_board_cards = sum(self.ns_board_cards)
        card_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n_ranks), gym.spaces.Discrete(self.n_suits))
        )
        hole_card_space = gym.spaces.Tuple((card_space,) * self.n_hole_cards)
        legal_actions = {}
        legal_actions['fold'] = gym.spaces.Discrete(max_action)
        legal_actions['check'] = gym.spaces.Discrete(max_action)
        legal_actions['call'] = gym.spaces.Discrete(max_action)
        n_splits = len(self.splits)
        for i in range(n_splits):
            legal_actions[f'raise_{i+1}'] = gym.spaces.Discrete(max_action)
        legal_actions['push'] = gym.spaces.Discrete(max_action)

        out = gym.spaces.Dict(
            {
                "action": gym.spaces.Discrete(max_action),
                "alive": gym.spaces.MultiBinary(self.n_players),
                "button": gym.spaces.Discrete(self.n_players),
                "call": gym.spaces.Discrete(max_action),
                "board_cards": gym.spaces.Tuple((card_space,) * n_board_cards),
                "hole_cards": gym.spaces.Tuple((hole_card_space,) * self.n_players),
                "max_raise": gym.spaces.Discrete(max_action),
                "min_raise": gym.spaces.Discrete(max_action),
                "pot": gym.spaces.Discrete(max_action),
                "stacks": gym.spaces.Tuple((gym.spaces.Discrete(max_action),) * self.n_players),
                "street_commits": gym.spaces.Tuple(
                    (gym.spaces.Discrete(max_action),) * self.n_players
                ),
                "legal_action": gym.spaces.Dict(legal_actions)
            }
        )
        return out

    @property
    def action_space(self):
        return gym.spaces.Discrete(max_action)

