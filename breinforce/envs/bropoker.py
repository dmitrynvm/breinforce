"""Bropoker environment class for running poker games"""
from datetime import datetime
import gym
import numpy as np
from typing import Dict, List, Optional, Tuple
from breinforce.agents import BaseAgent
from breinforce.games.bropoker import Deck, Judge
from breinforce.views import AsciiView


def clean(legal_actions, action) -> int:
    """
    Find closest bet size to actual bet
    """
    index = np.argmin(np.absolute(np.array(legal_actions) - action))
    return legal_actions[index]



class Bropoker(gym.Env):

    def __init__(
            self,
            n_players: int,
            n_streets: int,
            n_suits: int,
            n_ranks: int,
            n_hole_cards: int,
            n_cards_for_hand: int,
            rake: float,
            raise_sizes: List[float],
            n_board_cards: List[int],
            blinds: List[int],
            antes: List[int],
            stacks: List[int],
            splits: List[str]
    ) -> None:

        # envs
        self.n_players = n_players
        self.n_streets = n_streets
        self.n_suits = n_suits
        self.n_ranks = n_ranks
        self.n_hole_cards = n_hole_cards
        self.n_cards_for_hand = n_cards_for_hand
        self.rake = rake
        self.raise_sizes = raise_sizes
        self.n_board_cards = n_board_cards
        self.blinds = np.array(blinds, dtype=int)
        self.antes = np.array(antes, dtype=int)
        self.splits = splits

        # agnt
        self.start_stacks = np.array(stacks, dtype=int)
        self.small_blind = blinds[0]
        self.big_blind = blinds[1] if n_players > 2 else None
        self.straddle = blinds[2] if n_players > 3 else None
        self.hand_id = "hand_1"
        self.date = datetime.now().strftime("%b/%d/%Y %H:%M:%S")
        self.table_name = "table_1"
        self.player_ids = ["agent_" + str(i+1) for i in range(self.n_players)]
        self.hole_cards = []
        self.board_cards = []
        self.payouts = None

        # dealer
        self.street = 0
        self.button = 0
        self.player = -1
        self.pot = 0
        self.largest_raise = 0
        self.active = np.zeros(self.n_players, dtype=np.uint8)
        self.contribs = np.zeros(self.n_players, dtype=np.int32)
        self.stacks = np.array(stacks, dtype=np.int16)
        self.acted = np.zeros(self.n_players, dtype=np.uint8)
        self.commits = np.zeros(self.n_players, dtype=np.int32)
        self.history = []

        self.deck = Deck(self.n_suits, self.n_ranks)
        self.judge = Judge(n_suits, n_ranks, n_cards_for_hand)
        self.agents = None

    def reset(self) -> Dict:
        """Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        self.active.fill(1)
        self.stacks = self.start_stacks.copy()
        self.button = 0
        self.deck.shuffle()
        self.board_cards = self.deck.draw(sum(self.n_board_cards))
        self.history = []
        self.hole_cards = [
            self.deck.draw(self.n_hole_cards) for _
            in range(self.n_players)
        ]
        self.largest_raise = self.straddle
        self.pot = 0
        self.contribs.fill(0)
        self.street = 0
        self.commits.fill(0)
        self.acted.fill(0)
        self.player = self.button
        if self.n_players > 2:
            self.move()
        
        self.collect_antes()
        self.collect_blinds()
        self.move()
        self.move()
        return self.observation

    def act(self, obs: dict) -> int:
        return self.agents[self.player].act(obs)

    def action_type(self, action):
        call = self.call()
        max_raise = self.max_raise()

        out = None
        if action == call and call == 0:
            out = "check"
        elif action == call and call > 0:
            out = "call"
        elif call < action < max_raise:
            out = "raise"
        elif action == max_raise:
            out = "all_in"
        else:
            out = "fold"
        return out

    def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        legal_actions = self.legal_actions()
        action = clean(legal_actions, action)
        action_type = self.action_type(action)
        info = {
            "action_type": action_type,
            "legal_actions": legal_actions,
            "lower": self.call(),
            "upper": action - self.call()
        }

        # only fold if player cannot check
        if self.call() and ((action < self.call()) or action < 0):
            self.active[self.player] = 0
            action = 0
        # if action is full raise record as largest raise
        if action and (action - self.call()) >= self.largest_raise:
            self.largest_raise = action - self.call()
        self.__apply_one(action)

        self.history.append((self.state, self.player, action, info))
        self.acted[self.player] = True
        self.move()

        # if all agreed go to next street
        if self.__all_agreed():
            self.player = self.button
            self.move()
            # if at most 1 player active and not all in turn up all
            # board cards and evaluate hand
            while True:
                self.street += 1
                full_streets = self.street >= self.n_streets
                allin = self.active * (self.stacks == 0)
                all_allin = sum(self.active) - sum(allin) <= 1
                if full_streets:
                    break
                self.board_cards += self.deck.draw(
                    self.n_board_cards[self.street]
                )
                if not all_allin:
                    break
            self.commits.fill(0)
            self.acted = np.logical_not(self.active).astype(np.uint8)

        obs, payouts, done, _ = self.__output()
        self.payouts = payouts
        if all(done):
            self.player = -1
            obs["player"] = -1
            """
            if self.rake != 0.0:
                payouts = [
                    int(p * (1 - self.rake)) if p > 0 else p for p in payouts
                ]
            """
        obs["hole_cards"] = obs["hole_cards"][obs["player"]]
        return obs, payouts, done, info

    def __all_agreed(self) -> bool:
        if not all(self.acted):
            return False
        return all(
            (self.commits == self.commits.max())
            | (self.stacks == 0)
            | np.logical_not(self.active)
        )

    def __apply_one(self, action: int):
        action = min(self.stacks[self.player], action)
        self.pot += action
        self.contribs[self.player] += action
        self.commits[self.player] += action
        self.stacks[self.player] -= action


    def __done(self) -> List[bool]:
        if self.street >= self.n_streets or sum(self.active) <= 1:
            return np.full(self.n_players, 1)
        return np.logical_not(self.active)

    def __payouts(self) -> np.ndarray:
        # players that have folded lose their actions
        payouts = -1 * self.contribs * np.logical_not(self.active)
        if sum(self.active) == 1:
            payouts += self.active * (self.pot - self.contribs)
        # if last street played and still multiple players active
        elif self.street >= self.n_streets:
            payouts = self.__eval_round()
            payouts -= self.contribs
        if any(payouts > 0):
            self.stacks += payouts + self.contribs
        return payouts

    def __output(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        observation = self.observation
        payouts = self.__payouts()
        done = self.__done()
        return observation, payouts, done, None

    def __eval_round(self) -> np.ndarray:
        # grab array of hand strength and pot contribs
        worst_hand = self.judge.hashmap.max_rank + 1
        hand_list = []
        payouts = np.zeros(self.n_players, dtype=int)
        for player in range(self.n_players):
            # if not active hand strength set
            # to 1 worse than worst possible rank
            hand_strength = worst_hand
            if self.active[player]:
                hand_strength = self.judge.evaluate(
                    self.hole_cards[player], self.board_cards
                )
            hand_list.append([player, hand_strength, self.contribs[player]])
        hands = np.array(hand_list)
        # sort hands by hand strength and pot contribs
        hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
        pot = self.pot
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
            button_shift = (involved_players <= self.button) * self.n_players
            button_shifted_players = involved_players + button_shift
            worst_idx = np.argmin(button_shifted_players)
            worst_pos = involved_players[worst_idx]
            payouts[worst_pos] += remainder
        return payouts

    def move(self):
        for idx in range(1, self.n_players + 1):
            player = (self.player + idx) % self.n_players
            if self.active[player]:
                break
            else:
                self.acted[player] = True
        self.player = player

    def register(self, agents: List) -> None:
        self.agents = agents

    def legal_actions(self):
        legal_actions = {0, self.call(), self.max_raise()}
        for split in self.splits:
            legal_action = round(split * self.pot)
            if legal_action < self.max_raise():
                legal_actions.add(legal_action)
        return list(sorted(legal_actions))

    @property
    def observation(self) -> Dict:
        board_cards = [str(c) for c in self.board_cards]
        hole_cards = [[str(c) for c in cs] for cs in self.hole_cards]
        obs: dict = {
            "button": self.button,
            "player": self.player,
            "pot": self.pot,
            "call": self.call(),
            "max_raise": self.max_raise(),
            "min_raise": self.min_raise(),
            "legal_actions": self.legal_actions(),
            "board_cards": board_cards,
            "hole_cards": hole_cards,
            "active": self.active,
            "stacks": self.stacks,
            "commits": self.commits
        }
        return obs

    def call(self) -> int:
        output = 0
        if all(self.__done()):
            output = 0
        else:
            output = self.commits.max() - self.commits[self.player]
            output = min(output, self.stacks[self.player])
        return int(output)

    def min_raise(self) -> int:
        output = None
        if all(self.__done()):
            output = 0
        else:
            output = max(2 * self.straddle, self.largest_raise + self.call())
            output = min(output, self.stacks[self.player])
        return int(output)

    def max_raise(self) -> int:
        output = None
        if all(self.__done()):
            output = 0
        else:
            output = self.stacks[self.player]
            output = min(output, self.stacks[self.player])
        return int(output)

    def collect_antes(self):
        actions = self.antes
        actions = np.roll(actions, self.player)
        actions = (self.stacks > 0) * self.active * actions
        self.pot += sum(actions)
        self.contribs += actions
        self.stacks -= actions

    def collect_blinds(self):
        actions = self.blinds
        actions = np.roll(actions, self.player)
        actions = (self.stacks > 0) * self.active * actions
        self.pot += sum(actions)
        self.commits += actions
        self.contribs += actions
        self.stacks -= actions

    def render(self, mode="ascii"):
        return "ascii"

    @property
    def state(self):
        out = {
            "table_name": self.table_name,
            "street": self.street,
            "sb": self.small_blind,
            "bb": self.big_blind,
            "st": self.straddle,
            "hand_id": self.hand_id,
            "date": self.date,
            "player_ids": self.player_ids,
            "hole_cards": self.hole_cards,
            "start_stacks": self.start_stacks,
            "player": self.player,
            "allin": self.active * (self.stacks == 0),
            "board_cards": self.board_cards,
            "button": self.button,
            "done": all(self.__done()),
            "pot": self.pot,
            "payouts": self.payouts,
            "n_players": self.n_players,
            "n_hole_cards": self.n_hole_cards,
            "n_board_cards": sum(self.n_board_cards),
            "rake": self.rake,
            "antes": self.antes,
            "min_raise": self.min_raise(),
            "max_raise": self.max_raise(),
            "acted": self.acted.copy(),
            "active": self.active.copy(),
            "stacks": self.stacks.copy(),
            "call": self.call(),
            "commits": self.commits
        }
        return out
