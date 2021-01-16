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

# def __move_action(self):
#     for idx in range(1, self.n_players + 1):
#         player = (self.player + idx) % self.n_players
#         if self.active[player]:
#             break
#         else:
#             self.acted[player] = True
#     self.player = player


def reducer(state, action):
    state = deepcopy(state)
    if action["type"] == "DEAL_COMMUNITY_CARDS":
        state["community_cards"] = action["community_cards"]
    elif action["type"] == "DEAL_HOLE_CARDS":
        for i, player in enumerate(state["players"]):
            player["hole_cards"] = action["hole_cards"][i]
    elif action["type"] == "APPLY_ANTES":
        for i, player in enumerate(state["players"]):
            state["pot"] += state["antes"][i]
            player["contributed"] += state["antes"][i]
            player["stack"] -= state["antes"][i]
    elif action["type"] == "APPLY_BLINDS":
        for i, player in enumerate(state["players"]):
            state["pot"] += state["blinds"][i]
            player["contributed"] += state["blinds"][i]
            player["committed"] += state["blinds"][i]
            player["stack"] -= state["blinds"][i]
    return state


def create_init_state(config):
    output = {
        # constant
        "hand_id": "hand_1",
        "table_name": "table_1",
        "date": datetime.now().strftime("%b/%d/%Y %H:%M:%S"),
        "n_players": config['n_players'],
        "n_streets": config['n_streets'],
        "n_suits": config['n_suits'],
        "n_ranks": config['n_ranks'],
        "n_hole_cards": config['n_hole_cards'],
        "n_cards_for_hand": config['n_cards_for_hand'],
        "rake": config['rake'],
        "antes": config['antes'],
        "blinds": config['blinds'],
        "n_community_cards": config['n_community_cards'],
        "start_stacks": config['start_stacks'],
        "pot_splits": config['pot_splits'],
        # variable
        "street": 0,
        "button": 0,
        "player": 0,
        "pot": 0,
        "largest_raise": config['blinds'][2],
        "community_cards": [],
        "players": [
            {
                "id": "agent_1",
                "hole_cards": None,
                "stack": config['start_stacks'][0],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
            {
                "id": "agent_2",
                "hole_cards": None,
                "stack": config['start_stacks'][1],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
            {
                "id": "agent_3",
                "hole_cards": None,
                "stack": config['start_stacks'][2],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
            {
                "id": "agent_4",
                "hole_cards": None,
                "stack": config['start_stacks'][3],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
            {
                "id": "agent_5",
                "hole_cards": None,
                "stack": config['start_stacks'][4],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
            {
                "id": "agent_6",
                "hole_cards": None,
                "stack": config['start_stacks'][5],
                "alive": 1,
                "acted": 0,
                "committed": 0,
                "contributed": 0,
            },
        ]
    }
    return output


class Bropoker(gym.Env):

    def __init__(
            self, config
    ) -> None:
        n_suits = config['n_suits']
        n_ranks = config['n_ranks']
        n_cards_for_hand = config['n_cards_for_hand']
        self.config = config
        init_state = create_init_state(config)
        self.store = pydux.create_store(reducer, init_state)
        self.history = []
        self.agents = []
        self.deck = Deck(n_suits, n_ranks)
        self.judge = Judge(n_suits, n_ranks, n_cards_for_hand)

    def register(self, agents: List[BaseAgent]) -> None:
        self.agents = agents

    def reset(self) -> Dict:
        """Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        init_state = create_init_state(self.config)
        self.store = pydux.create_store(reducer, init_state)
        self.history = []
        self.deck.shuffle()

        n_comm_cards = self.store.get_state()['n_community_cards']
        n_community_cards = sum(n_comm_cards)
        self.store.dispatch({
            "type": "DEAL_COMMUNITY_CARDS",
            "community_cards": self.deck.draw(n_community_cards)
        })

        n_players = self.store.get_state()['n_players']
        n_hole_cards = self.store.get_state()['n_hole_cards']
        hole_cards = [
            self.deck.draw(n_hole_cards) for _ in range(n_players)
        ]
        self.store.dispatch({
            "type": "DEAL_HOLE_CARDS",
            "hole_cards": hole_cards
        })

        self.store.dispatch({
            "type": "APPLY_ANTES",
        })

        self.store.dispatch({
            "type": "APPLY_BLINDS",
        })

        # # in heads up button posts small blind
        # if self.n_players > 2:
        #     self.__move_action()

        # self.__apply_many(self.antes, False)
        # self.__apply_many(self.blinds, True)
        # self.__move_action()
        # self.__move_action()
        # return self.observation

    # def act(self, obs: dict) -> int:
    #     return self.agents[self.player].act(obs)

    # def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
    #     """Advances poker game to next player. If the action is 0, it is
    #     either considered a check or fold, depending on the previous
    #     action. The given bet is always rounded to the closest valid bet
    #     size. When it is the same distance from two valid bet sizes
    #     the smaller action size is used, e.g. if the min raise is 10 and
    #     the bet is 5, it is rounded down to 0.
    #     action = round(action)
    #     """

    #     call, min_raise, max_raise = self.__action_sizes()
    #     action = self.__clean_action(action, call, min_raise, max_raise)

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
    #         # community cards and evaluate hand
    #         while True:
    #             self.street += 1
    #             full_streets = self.street >= self.n_streets
    #             allin = self.active * (self.stacks == 0)
    #             all_allin = sum(self.active) - sum(allin) <= 1
    #             if full_streets:
    #                 break
    #             self.community_cards += self.deck.draw(
    #                 self.n_community_cards[self.street]
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

    # def __action_sizes(self) -> Tuple[int, int, int]:
    #     if all(self.__done()):
    #         call = min_raise = max_raise = 0
    #     else:
    #         call = self.committed.max() - self.committed[self.player]
    #         call = min(call, self.stacks[self.player])
    #         min_raise = max(2 * self.straddle, self.largest_raise + call)
    #         min_raise = min(min_raise, self.stacks[self.player])
    #         max_raise = self.stacks[self.player]
    #         max_raise = min(max_raise, self.stacks[self.player])
    #     return call, min_raise, max_raise

    # @staticmethod
    # def __clean_action(
    #     action: int,
    #     call: int,
    #     min_raise: int,
    #     max_raise: int
    # ) -> int:
    #     # find closest action size to actual action
    #     # pessimistic approach: in ties order is fold/check -> call -> raise
    #     lst = [0, call, min_raise, max_raise]
    #     idx = np.argmin(np.absolute(np.array(lst) - action))
    #     # if call closest
    #     if idx == 1:
    #         return call
    #     # if min raise or max raise closest
    #     if idx in (2, 3):
    #         return round(min(max_raise, max(min_raise, action)))
    #     # if fold closest
    #     return 0

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

    # def __done(self) -> List[bool]:
    #     if self.street >= self.n_streets or sum(self.active) <= 1:
    #         return np.full(self.n_players, 1)
    #     return np.logical_not(self.active)

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
    #                 self.hole_cards[player], self.community_cards
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

    # @property
    # def observation(self) -> Dict:
    #     community_cards = [str(c) for c in self.community_cards]
    #     hole_cards = [[str(c) for c in cs] for cs in self.hole_cards]
    #     obs: dict = {
    #         "button": self.button,
    #         "player": self.player,
    #         "pot": self.pot,
    #         "call": self.call,
    #         "max_raise": self.max_raise,
    #         "min_raise": self.min_raise,
    #         "legal_actions": self.legal_actions,
    #         "community_cards": community_cards,
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
    #         "community_cards": self.community_cards,
    #         "button": self.button,
    #         "done": all(self.__done()),
    #         "pot": self.pot,
    #         "payouts": self.payouts,
    #         "n_players": self.n_players,
    #         "n_hole_cards": self.n_hole_cards,
    #         "n_community_cards": sum(self.n_community_cards),
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
    # def call(self) -> int:
    #     output = 0
    #     if all(self.__done()):
    #         output = 0
    #     else:
    #         output = self.committed.max() - self.committed[self.player]
    #         output = min(output, self.stacks[self.player])
    #     return int(output)

    # @property
    # def min_raise(self) -> int:
    #     output = None
    #     if all(self.__done()):
    #         output = 0
    #     else:
    #         output = max(2 * self.straddle, self.largest_raise + self.call)
    #         output = min(output, self.stacks[self.player])
    #     return int(output)

    # @property
    # def max_raise(self) -> int:
    #     output = None
    #     if all(self.__done()):
    #         output = 0
    #     else:
    #         output = self.stacks[self.player]
    #         output = min(output, self.stacks[self.player])
    #     return int(output)

    # @property
    # def json_state(self):
    #     player = self.player
    #     player_id = self.player_ids[player]
    #     community_cards = str([repr(c) for c in self.community_cards])
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
    #         "community_cards": community_cards,
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

    def render(self, mode="dict"):
        out = None
        if mode == 'dict':
            out = pprint.pformat(self.store.get_state(), sort_dicts=False)
        return out
