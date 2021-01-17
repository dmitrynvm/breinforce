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

    # @property
    # def legal_actions(self):
    #     bets = set()
    #     # fold/check
    #     bets.add(0)
    #     # call
    #     bets.add(self.call)
    #     # raises
    #     for split in self.pot_splits:
    #         bet = round(split * self.pot)
    #         if bet < self.max_raise:
    #             bets.add(bet)
    #     # all_in
    #     bets.add(self.max_raise)
    #     return list(sorted(bets))


# getters

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

# @property
# def max_raise(self) -> int:
#     output = None
#     if all(self.__done()):
#         output = 0
#     else:
#         output = self.stacks[self.player]
#         output = min(output, self.stacks[self.player])
#     return int(output)


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

def move_turn():
    return {"type": "MOVE_TURN"}

def update_legal_actions(state, call, min_raise, max_raise):
    return {
        "type": "UPDATE_LEGAL_ACTIONS",
        "call": call,
        "min_raise": min_raise,
        "max_raise": max_raise
    }

# reducers
def engine(state, action):
    global deck, judge
    state = deepcopy(state)
    if action["type"] == "@@redux/INIT":
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
    elif action["type"] == "MOVE_TURN":
        for i in range(state["n_players"]):
            seat = (state["player"] + i + 1) % state['n_players']
            player = state["players"][seat]
            if player["alive"]:
                break
            else:
                player["acted"] = True
        state["player"] = seat
    elif action["type"] == "UPDATE_LEGAL_ACTIONS":
        legal_actions = {0, action["call"], action["max_raise"]}
        for split in state["splits"]:
            legal_action = round(split * state["pot"])
            if legal_action < action["max_raise"]:
                legal_actions.add(legal_action)
        print(list(sorted(legal_actions)))
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
                "legal_actions": [],
                "hole_cards": [],
            },
            {
                "id": "agent_2",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "legal_actions": [],
                "hole_cards": [],
            },
            {
                "id": "agent_3",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "legal_actions": [],
                "hole_cards": [],
            },
            {
                "id": "agent_4",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "legal_actions": [],
                "hole_cards": [],
            },
            {
                "id": "agent_5",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "legal_actions": [],
                "hole_cards": [],
            },
            {
                "id": "agent_6",
                "alive": True,
                "acted": False,
                "commit": 0,
                "contrib": 0,
                "stack": 0,
                "legal_actions": [],
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
            self.dispatch(move_turn())
        self.dispatch(move_turn())
        self.dispatch(move_turn())
        return {}

    def act(self, obs: dict) -> int:
        return self.agents[self.player].act(obs)

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

    def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Advances poker game to next player. If the action is 0, it is
        either considered a check or fold, depending on the previous
        action. The given bet is always rounded to the closest valid bet
        size. When it is the same distance from two valid bet sizes
        the smaller action size is used, e.g. if the min raise is 10 and
        the bet is 5, it is rounded down to 0.
        action = round(action)
        """

        call = get_call(self.state)
        min_raise = get_min_raise(self.state)
        max_raise = get_max_raise(self.state)
        self.dispatch(update_legal_actions(self.state, call, min_raise, max_raise))

    #     # only fold if player cannot check
    #     if call and ((action < call) or action < 0):
    #         self.active[self.player] = 0
    #         action = 0
    #     # if action is full raise record as largest raise
    #     if action and (action - call) >= self.largest_raise:
    #         self.largest_raise = action - call
    #     self.__apply_one(action)

    #     self.history.append((self.state, self.player+1, action))
    #     self.acted[self.player] = True
    #     self.__move_action()

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

    # def __all_agreed(self) -> bool:
    #     if not all(self.acted):
    #         return False
    #     return all(
    #         (self.committed == self.committed.max())
    #         | (self.stacks == 0)
    #         | np.logical_not(self.active)
    #     )

    # def __apply_one(self, action: int):
    #     action = min(self.stacks[self.player], action)
    #     self.pot += action
    #     self.pot_commit[self.player] += action
    #     self.committed[self.player] += action
    #     self.stacks[self.player] -= action

    # def __apply_many(
    #     self,
    #     actions: List[int],
    #     is_committed: bool = True
    # ):
    #     actions = np.roll(actions, self.player)
    #     actions = (self.stacks > 0) * self.active * actions
    #     if is_committed:
    #         self.committed += actions
    #     self.pot_commit += actions
    #     self.pot += sum(actions)
    #     self.stacks -= actions


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

    # @property
    # def state(self):
    #     output = {
    #         # auxilary
    #         "table_name": self.table_name,
    #         "sb": self.small_blind,
    #         "bb": self.big_blind,
    #         "st": self.straddle,
    #         "hand_id": self.hand_id,
    #         "date1": self.date1,
    #         "date2": self.date2,
    #         "player_ids": self.player_ids,
    #         "hole_cards": self.hole_cards,
    #         "start_stacks": self.start_stacks,
    #         # config
    #         "player": self.player,
    #         "allin": self.active * (self.stacks == 0),
    #         "board_cards": self.board_cards,
    #         "button": self.button,
    #         "done": all(self.__done()),
    #         "pot": self.pot,
    #         "payouts": self.payouts,
    #         "n_players": self.n_players,
    #         "n_hole_cards": self.n_hole_cards,
    #         "n_board_cards": sum(self.n_board_cards),
    #         "rake": self.rake,
    #         "antes": self.antes,
    #         "street": self.street,
    #         "min_raise": self.min_raise,
    #         "max_raise": self.max_raise,
    #         "acted": self.acted.copy(),
    #         "active": self.active.copy(),
    #         "stacks": self.stacks.copy(),
    #         "call": self.call,
    #         "committed": self.committed
    #     }
    #     return output

    # @property
    # def json_state(self):
    #     player = self.player
    #     player_id = self.player_ids[player]
    #     board_cards = str([repr(c) for c in self.board_cards])
    #     hole_cards = str([repr(c) for c in self.hole_cards[player]])
    #     legal_actions = str([int(a) for a in self.legal_actions])

    #     out = {
    #         "hand": self.hand_id,
    #         "street": self.street + 1,
    #         "button": self.button + 1,
    #         "player": player + 1,
    #         "player_id": player_id,
    #         "pot": int(self.pot),
    #         "call": self.call,
    #         "min_raise": self.min_raise,
    #         "max_raise": self.max_raise,
    #         "active": bool(self.active[self.player]),
    #         "board_cards": board_cards,
    #         "hole_cards": hole_cards,
    #         "legal_actions": legal_actions
    #     }

    #     return out

    # @property
    # def json_history(self):
    #     return {
    #         "0": {
    #         }
    #     }

