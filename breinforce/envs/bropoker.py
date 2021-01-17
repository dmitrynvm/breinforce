"""Bropoker environment class for running poker games"""
from datetime import datetime
import json
import gym
import numpy as np
import os
import uuid
import pprint
import pydux
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from breinforce import errors
from breinforce.agents import BaseAgent
from breinforce.games.bropoker import Card, Deck, Judge

# def __all_agreed(self) -> bool:
#     if not all(self.acted):
#         return False
#     return all(
#         (self.committed == self.committed.max())
#         | (self.stacks == 0)
#         | np.logical_not(self.active)
#     )

# @property
# def observation(self) -> Dict:
#     board_cards = [str(c) for c in self.board_cards]
#     hole_cards = [[str(c) for c in cs] for cs in self.hole_cards]
#     obs: dict = {
#         "button": self.button,
#         "player": self.player,
#         "pot": self.pot,
#         "call": self.call,
#         "max_raise": self.max_raise,
#         "min_raise": self.min_raise,
#         "legal_actions": self.legal_actions,
#         "board_cards": board_cards,
#         "hole_cards": hole_cards,
#         "active": self.active,
#         "stacks": self.stacks,
#         "committed": self.committed
#     }
#     return obs


# getters

def clean(state: dict, action: int) -> int:
    """
    Find closest bet size to actual bet
    """
    legal_actions = get_legal_actions(state)
    index = np.argmin(np.absolute(np.array(legal_actions) - action))
    return legal_actions[index]

def get_small_blind(state):
    return state["blinds"][0]

def get_big_blind(state):
    return state["blinds"][1] if state["n_players"] > 2 else state["blinds"][0]

def get_straddle(state):
    return state["blinds"][2] if state["n_players"] > 3 else state["blinds"][1]

def get_done(state):
    out = None
    if state["street"] >= state["n_streets"]:
        out = [True] * state["n_players"]
    out = [not player["alive"] for player in state["players"]]
    return out

def get_player(state):
    return state["players"][state["player"]]

def get_call(state):
    out = 0
    done = get_done(state)
    if not all(done):
        commits = [player["commit"] for player in state["players"]]
        max_commit = max(commits)
        for player in state["players"]:
            out = min(max_commit - player["commit"], player["stack"])
    return out

def get_min_raise(state):
    out = 0
    player = get_player(state)
    done = get_done(state)
    call = get_call(state)
    straddle = get_straddle(state)
    max_commit = sum([player["commit"] for player in state["players"]])
    if not all(done):
        out = max(2 * straddle, max_commit + call)
        out = min(out, player["stack"])
    return out

def get_max_raise(state):
    out = 0
    player = state["players"][state["player"]]
    done = get_done(state)
    call = get_call(state)

    straddle = get_straddle(state)
    max_commit = sum([player["commit"] for player in state["players"]])
    if not all(done):
        out = max(2 * straddle, player["stack"])
    return out

def get_legal_actions(state):
    call = get_call(state)
    max_raise = get_max_raise(state)
    legal_actions = {0, call, max_raise}
    player = get_player(state)
    for split in state["splits"]:
        legal_action = round(split * state["pot"])
        if legal_action < max_raise:
            legal_actions.add(legal_action)
    return list(sorted(legal_actions))

def observe(state):
    out = {}
    player = get_player(state)
    out["legal_actions"] = get_legal_actions(state)
    return out

def make_fold(state):
    player = get_player(state)
    player['alive'] = False
    return list(sorted(legal_actions))


# objects
deck = None
judge = None

# action creators
def shuffle_deck():
    return {"type": "SHUFFLE_DECK"}

def deal_board_cards():
    return {"type": "DEAL_BOARD_CARDS"}

def deal_hole_cards():
    return {"type": "DEAL_HOLE_CARDS"}

def init_stacks():
    return {"type": "INIT_STACKS"}

def collect_antes():
    return {"type": "COLLECT_ANTES"}

def collect_blinds():
    return {"type": "COLLECT_BLINDS"}

def move():
    return {"type": "MOVE"}

def perform(amount):
    return {
        "type": "PERFORM",
        "amount": amount
    }

# reducers
def engine(state, action):
    state = deepcopy(state)
    if action["type"] == "@@redux/INIT":
        global deck, judge
        deck = Deck(state["n_suits"], state["n_ranks"])
        judge = Judge(
            state["n_suits"],
            state["n_ranks"],
            state["n_cards_for_hand"]
        )
    elif action["type"] == "SHUFFLE_DECK":
        deck.shuffle()
    elif action["type"] == "DEAL_BOARD_CARDS":
        state["board_cards"] = deck.draw(sum(state["n_board_cards"]))
    elif action["type"] == "DEAL_HOLE_CARDS":
        for i, player in enumerate(state["players"]):
            player["hole_cards"] = deck.draw(state["n_hole_cards"])
    elif action["type"] == "INIT_STACKS":
        for i, player in enumerate(state["players"]):
            player["stack"] = state["stacks"][i]
    elif action["type"] == "COLLECT_ANTES":
        for i, player in enumerate(state["players"]):
            ante = state["antes"][i]
            if ante:
                state["pot"] += ante
                player["contrib"] += ante
                player["stack"] -= ante
    elif action["type"] == "COLLECT_BLINDS":
        for i, player in enumerate(state["players"]):
            blind = state["blinds"][i]
            if blind:
                state["pot"] += blind
                player["acted"] = True
                player["contrib"] += blind
                player["commit"] += blind
                player["stack"] -= blind
    elif action["type"] == "MOVE":
        for i in range(state["n_players"]):
            seat = (state["player"] + i + 1) % state['n_players']
            player = state["players"][seat]
            if player["alive"]:
                break
            else:
                player["acted"] = True
        state["player"] = seat
    elif action["type"] == "UPDATE_LEGAL_ACTIONS":
        state["legal_actions"] = get_legal_actions(state)
    elif action["type"] == "PERFORM":
        player = get_player(state)
        amount = action['amount']
        call = get_call(state)
        if amount > call:
            state["pot"] += amount
            player["contrib"] += amount
            player["commit"] += amount
            player["stack"] -= amount
            player["acted"] = True
        else:
            player['alive'] = False
    else:
        print("Unknown action", action)
    return state


def create_init_state(config):
    output = {
        **config,
        "hand_id": "hand_1",
        "table_name": "table_1",
        "date": datetime.now().strftime("%b/%d/%Y %H:%M:%S"),
        "street": 0,
        "button": 0,
        "player": 0,
        "pot": 0,
        "board_cards": [],
        "players": [
            {
                "id": "agent_1",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
            {
                "id": "agent_2",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
            {
                "id": "agent_3",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
            {
                "id": "agent_4",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
            {
                "id": "agent_5",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
            {
                "id": "agent_6",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "hole_cards": [],
            },
        ]
    }
    return output

def create_store(config):
    return pydux.create_store(engine, create_init_state(config))


class Bropoker(gym.Env):

    def __init__(self, config) -> None:
        self.config = config
        self.agents = []
        self.store = None
        self.history = []

    def register(self, agents: List[BaseAgent]) -> None:
        self.agents = agents

    def reset(self) -> Dict:
        """
        Resets the environment to an initial state and returns an initial
        observation.
        Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        self.store = create_store(self.config)
        self.history = []
        self.dispatch(shuffle_deck())
        self.dispatch(deal_board_cards())
        self.dispatch(deal_hole_cards())
        self.dispatch(init_stacks())
        self.dispatch(collect_antes())
        self.dispatch(collect_blinds())
        if self.state["n_players"] > 2:
            self.dispatch(move())
        self.dispatch(move())
        self.dispatch(move())
        return observe(self.state)

    def act(self, obs: dict) -> int:
        player = self.state['player']
        return self.agents[player].act(obs)

    @property
    def dispatch(self):
        return self.store.dispatch

    @property
    def state(self):
        return self.store.get_state()

    def render(self, mode="dict"):
        out = None
        if mode == 'dict':
            out = pprint.pformat(self.store.get_state(), sort_dicts=False)
        return out

    def step(self, amount: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Advances poker game to next player. If the action is 0, it is
        either considered a check or fold, depending on the previous
        action. The given bet is always rounded to the closest valid bet
        size. When it is the same distance from two valid bet sizes
        the smaller action size is used, e.g. if the min raise is 10 and
        the bet is 5, it is rounded down to 0.
        action = round(action)
        """
        amount = clean(self.state, amount)
        self.dispatch(perform(amount))
        self.history.append(self.state)
        self.dispatch(move())
        return (1, 2, [3], 4)

    #     # if all agreed go to next street
    #     if self.__all_agreed():
    #         self.player = self.button
    #         self.__move_action()
    #         # if at most 1 player active and not all in turn up all
    #         # board cards and evaluate hand
    #         while True:
    #             self.street += 1
    #             full_streets = self.street >= self.n_streets
    #             allin = self.active * (self.stacks == 0)
    #             all_allin = sum(self.active) - sum(allin) <= 1
    #             if full_streets:
    #                 break
    #             self.board_cards += self.deck.draw(
    #                 self.n_board_cards[self.street]
    #             )
    #             if not all_allin:
    #                 break
    #         self.committed.fill(0)
    #         self.acted = np.logical_not(self.active).astype(np.uint8)

    #     obs, payouts, done, _ = self.__output()
    #     self.payouts = payouts
    #     if all(done):
    #         self.player = -1
    #         obs["player"] = -1
    #         """
    #         if self.rake != 0.0:
    #             payouts = [
    #                 int(p * (1 - self.rake)) if p > 0 else p for p in payouts
    #             ]
    #         """
    #     obs["hole_cards"] = obs["hole_cards"][obs["player"]]
    #     return obs, payouts, done, None

    # def __payouts(self) -> np.ndarray:
    #     # players that have folded lose their actions
    #     payouts = -1 * self.pot_commit * np.logical_not(self.active)
    #     if sum(self.active) == 1:
    #         payouts += self.active * (self.pot - self.pot_commit)
    #     # if last street played and still multiple players active
    #     elif self.street >= self.n_streets:
    #         payouts = self.__eval_round()
    #         payouts -= self.pot_commit
    #     if any(payouts > 0):
    #         self.stacks += payouts + self.pot_commit
    #     return payouts

    # def __output(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
    #     observation = self.observation
    #     payouts = self.__payouts()
    #     done = self.__done()
    #     return observation, payouts, done, None

    # def __eval_round(self) -> np.ndarray:
    #     # grab array of hand strength and pot commits
    #     worst_hand = self.judge.hashmap.max_rank + 1
    #     hand_list = []
    #     payouts = np.zeros(self.n_players, dtype=int)
    #     for player in range(self.n_players):
    #         # if not active hand strength set
    #         # to 1 worse than worst possible rank
    #         hand_strength = worst_hand
    #         if self.active[player]:
    #             hand_strength = self.judge.evaluate(
    #                 self.hole_cards[player], self.board_cards
    #             )
    #         hand_list.append([player, hand_strength, self.pot_commit[player]])
    #     hands = np.array(hand_list)
    #     # sort hands by hand strength and pot commits
    #     hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
    #     pot = self.pot
    #     remainder = 0
    #     # iterate over hand strength and
    #     # pot commits from smallest to largest
    #     for idx, (_, strength, pot_commit) in enumerate(hands):
    #         eligible = hands[:, 0][hands[:, 1] == strength].astype(int)
    #         # cut can only be as large as lowest player commit amount
    #         cut = np.clip(hands[:, 2], None, pot_commit)
    #         split_pot = sum(cut)
    #         split = split_pot // len(eligible)
    #         remain = split_pot % len(eligible)
    #         payouts[eligible] += split
    #         remainder += remain
    #         # remove chips from players and pot
    #         hands[:, 2] -= cut
    #         pot -= split_pot
    #         # remove player from next split pot
    #         hands[idx, 1] = worst_hand
    #         if pot == 0:
    #             break
    #     # give worst position player remainder chips
    #     if remainder:
    #         # worst player is first player after button involved in pot
    #         involved_players = np.nonzero(payouts)[0]
    #         button_shift = (involved_players <= self.button) * self.n_players
    #         button_shifted_players = involved_players + button_shift
    #         worst_idx = np.argmin(button_shifted_players)
    #         worst_pos = involved_players[worst_idx]
    #         payouts[worst_pos] += remainder
    #     return payouts




