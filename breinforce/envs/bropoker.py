"""Bropoker environment class for running poker games"""
from datetime import datetime
import json
import gym
import numpy as np
import os
import uuid
from typing import Dict, List, Optional, Tuple, Union
from breinforce import exceptions
from breinforce.agents import BaseAgent
from breinforce.config.application import CONFIG_DIR
from breinforce.games.bropoker import Card, Deck, Judge
from breinforce.views import AsciiView
from . import utils


class Bropoker(gym.Env):
    """Runs a range of different of poker games dependent on the
    given configuration. Supports limit, no limit and pot limit
    bet sizing, arbitrary deck sizes, arbitrary hole and community
    cards and many other options.
    """

    @staticmethod
    def configure():
        """ Merges the local configured envs to the global OpenAI Gym list.
        """
        try:
            config_path = os.path.join(CONFIG_DIR, "envs.json")
            with open(config_path, "r") as f:
                configs = json.loads(f.read())
                for game in configs:
                    for name, config in configs[game].items():
                        configs[game][name] = utils.parse_config(config)
                        Bropoker.register(configs[game])
        except ImportError:
            pass

    @staticmethod
    def register(configs: Dict) -> None:
        """ Registers dict of breinforce configs as gym environments
        """
        env_entry_point = "breinforce.envs:Bropoker"
        env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]
        for env_id, config in configs.items():
            if env_id not in env_ids:
                gym.envs.registration.register(
                    id=env_id,
                    entry_point=env_entry_point,
                    kwargs={**config}
                )

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
            n_raises: List[float],
            n_community_cards: List[int],
            blinds: List[int],
            antes: List[int],
            start_stacks: List[int]
    ) -> None:

        # config
        self.n_players = n_players
        self.n_streets = n_streets
        self.n_suits = n_suits
        self.n_ranks = n_ranks
        self.n_hole_cards = n_hole_cards
        self.n_cards_for_hand = n_cards_for_hand
        self.rake = rake
        self.raise_sizes = [self.__clean(rs) for rs in raise_sizes]
        self.n_raises = [float(raise_num) for raise_num in n_raises]
        self.n_community_cards = n_community_cards
        self.blinds = np.array(blinds)
        self.antes = np.array(antes)

        # auxilary
        self.small_blind = blinds[0]
        self.big_blind = blinds[1] if n_players > 2 else None
        self.straddle = blinds[2] if n_players > 3 else None
        self.hand_name = "hand_1"
        self.date1 = datetime.now().strftime("%b/%d/%Y %H:%M:%S")
        self.date2 = datetime.now().strftime("%b/%d/%Y %H:%M:%S")
        self.table_name = "table_1"
        self.player_names = [
            "agent_" + str(i+1) for i in range(self.n_players)
        ]
        self.start_stacks = np.array(start_stacks)
        self.payouts = None
        self.call = None
        self.min_raise = None
        self.max_raise = None
        self.hole_cards = []
        self.community_cards = []
        self.deck = Deck(self.n_suits, self.n_ranks)

        # dealer
        self.street = 0
        self.button = 0
        self.player = -1
        self.pot = 0
        self.largest_raise = 0
        self.active = np.zeros(self.n_players, dtype=np.uint8)
        self.pot_commit = np.zeros(self.n_players, dtype=np.int32)
        self.stacks = np.array(start_stacks)
        self.acted = np.zeros(self.n_players, dtype=np.uint8)
        self.street_commits = np.zeros(self.n_players, dtype=np.int32)
        self.street_raises = 0
        self.history = []

        self.judge = Judge(n_suits, n_ranks, n_cards_for_hand)
        self.agents: Optional[Dict[int, BaseAgent]] = None

    def reset(self) -> Dict:
        """Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        self.active.fill(1)
        self.stacks = self.start_stacks.copy()
        self.button = 0
        self.deck.shuffle()
        self.community_cards = self.deck.draw(self.n_community_cards[0])
        self.history = []
        self.hole_cards = [
            self.deck.draw(self.n_hole_cards) for _
            in range(self.n_players)
        ]
        self.largest_raise = self.straddle
        self.pot = 0
        self.pot_commit.fill(0)
        self.street = 0
        self.street_commits.fill(0)
        self.acted.fill(0)
        self.street_raises = 0

        self.player = self.button
        # in heads up button posts small blind
        if self.n_players > 2:
            self.__move_action()

        self.__apply_many(actions=self.antes, street_commits=False)
        self.__apply_many(actions=self.blinds, street_commits=True)
        self.__move_action()
        self.__move_action()
        return self.__observation()

    def act(self, obs: dict) -> int:
        action = self.agents[self.player].act(obs)
        return action

    def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Advances poker game to next player. If the action is 0, it is
        either considered a check or fold, depending on the previous
        action. The given bet is always rounded to the closest valid bet
        size. When it is the same distance from two valid bet sizes
        the smaller action size is used, e.g. if the min raise is 10 and
        the bet is 5, it is rounded down to 0.

        Parameters
        ----------
        action : int
            number of chips bet by player currently active

        Returns
        -------
        Tuple[Dict, np.ndarray, np.ndarray]
            observation dictionary containing following info

                {
                    active: position of active player
                    button: position of button
                    call: number of chips needed to call
                    community_cards: shared community cards
                    hole_cards: hole cards for every player
                    max_raise: maximum raise size
                    min_raise: minimum raise size
                    pot: number of chips in the pot
                    stacks: stack size for every player
                    street_commits: number of chips commited by every
                                    player on this street
                }

            payouts for every player

            bool array containing value for every player if that player
            is still involved in round
        """
        if self.player == -1:
            if any(self.active):
                return self.__output()
            raise exceptions.HashMapResetError(
                "call reset() before calling first step()"
            )

        fold = action < 0
        action = round(action)

        call, min_raise, max_raise = self.__action_sizes()
        self.call = call
        self.min_raise = min_raise
        self.max_raise = max_raise
        # round action to nearest sizing
        action = self.__clean_action(action, call, min_raise, max_raise)

        # only fold if player cannot check
        if call and ((action < call) or fold):
            self.active[self.player] = 0
            action = 0
            called = True
        # if action is full raise record as largest raise
        checked = call == 0 and action == 0
        raised = call > 0 and action > 0
        if action and (action - call) >= self.largest_raise:
            self.largest_raise = action - call
            self.street_raises += 1
        self.__apply_one(action)
        action = int(action)
        action_type = None
        if action < 0:
            action_type = "fold"
        elif action == 0:
            action_type = "check"
        elif action == min_raise:
            action_type = "call"
        else:
            action_type = "raise"
        self.history.append((self.state, self.player+1, action, None))
        self.acted[self.player] = True
        self.__move_action()

        # if all agreed go to next street
        if self.__all_agreed():
            self.player = self.button
            self.__move_action()
            # if at most 1 player active and not all in turn up all
            # community cards and evaluate hand
            while True:
                self.street += 1
                full_streets = self.street >= self.n_streets
                allin = self.active * (self.stacks == 0)
                all_allin = sum(self.active) - sum(allin) <= 1
                if full_streets:
                    break
                self.community_cards += self.deck.draw(
                    self.n_community_cards[self.street]
                )
                if not all_allin:
                    break
            self.street_commits.fill(0)
            self.acted = np.logical_not(self.active).astype(np.uint8)
            self.street_raises = 0

        obs, payouts, done, info = self.__output()
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
        return obs, payouts, done, None

    def __all_agreed(self) -> bool:
        # not all agreed if not all players had chance to act
        if not all(self.acted):
            return False
        # all agreed if street commits equal to maximum street commit
        # or player is all in
        # or player is not active
        return all(
            (self.street_commits == self.street_commits.max())
            | (self.stacks == 0)
            | np.logical_not(self.active)
        )

    def __action_sizes(self) -> Tuple[int, int, int]:
        # call difference actionween commit and maximum commit
        call = self.street_commits.max() - self.street_commits[self.player]
        # min raise at least largest previous raise
        # if limit game min and max raise equal to raise size
        if isinstance(self.raise_sizes[self.street], int):
            max_raise = min_raise = self.raise_sizes[self.street] + call
        else:
            min_raise = max(self.straddle, self.largest_raise + call)
            if self.raise_sizes[self.street] == "pot":
                max_raise = self.pot + call * 2
            elif self.raise_sizes[self.street] == float("inf"):
                max_raise = self.stacks[self.player]
        # if maximum number of raises in street
        # was reached cap raise at 0
        if self.street_raises >= self.n_raises[self.street]:
            min_raise = max_raise = 0
        # if last full raise was done by active player
        # (another player has raised less than minimum raise amount)
        # cap active players raise size to 0
        if self.street_raises and call < self.largest_raise:
            min_raise = max_raise = 0
        # clip actions to stack size
        call = min(call, self.stacks[self.player])
        min_raise = min(min_raise, self.stacks[self.player])
        max_raise = min(max_raise, self.stacks[self.player])
        return call, min_raise, max_raise

    @staticmethod
    def __clean_action(
        action: int,
        call: int,
        min_raise: int,
        max_raise: int
    ) -> int:
        # find closest action size to actual action
        # pessimistic approach: in ties order is fold/check -> call -> raise
        lst = [0, call, min_raise, max_raise]
        idx = np.argmin(np.absolute(np.array(lst) - action))
        # if call closest
        if idx == 1:
            return call
        # if min raise or max raise closest
        if idx in (2, 3):
            return round(min(max_raise, max(min_raise, action)))
        # if fold closest
        return 0

    def __apply_one(self, action: int):
        # action only as large as stack size
        action = min(self.stacks[self.player], action)

        self.pot += action
        self.pot_commit[self.player] += action
        self.street_commits[self.player] += action
        self.stacks[self.player] -= action

    def __apply_many(
        self,
        actions: List[int],
        street_commits: bool = True
    ):
        actions = np.roll(actions, self.player)
        actions = (self.stacks > 0) * self.active * actions
        if street_commits:
            self.street_commits += actions
        self.pot_commit += actions
        self.pot += sum(actions)
        self.stacks -= actions

    def __done(self) -> List[bool]:
        if self.street >= self.n_streets or sum(self.active) <= 1:
            # end game
            out = np.full(self.n_players, 1)
            return out
        return np.logical_not(self.active)

    def __observation(self) -> Dict:
        if all(self.__done()):
            call = min_raise = max_raise = 0
        else:
            call, min_raise, max_raise = self.__action_sizes()
        community_cards = [str(card) for card in self.community_cards]
        hole_cards = [
            [str(card) for card in cards] for cards in self.hole_cards
        ]
        obs: dict = {
            "player": self.player,
            "active": self.active,
            "button": self.button,
            "call": call,
            "community_cards": community_cards,
            "hole_cards": hole_cards,
            "max_raise": max_raise,
            "min_raise": min_raise,
            "pot": self.pot,
            "stacks": self.stacks,
            "street_commits": self.street_commits,
        }
        return obs

    def __payouts(self) -> np.ndarray:
        # players that have folded lose their actions
        payouts = -1 * self.pot_commit * np.logical_not(self.active)
        if sum(self.active) == 1:
            payouts += self.active * (self.pot - self.pot_commit)
        # if last street played and still multiple players active
        elif self.street >= self.n_streets:
            payouts = self.__eval_round()
            payouts -= self.pot_commit
        if any(payouts > 0):
            self.stacks += payouts + self.pot_commit
        return payouts

    def __output(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        observation = self.__observation()
        payouts = self.__payouts()
        done = self.__done()
        return observation, payouts, done, None

    def __eval_round(self) -> np.ndarray:
        # grab array of hand strength and pot commits
        worst_hand = self.judge.hashmap.max_rank + 1
        hand_list = []
        payouts = np.zeros(self.n_players, dtype=int)
        for player in range(self.n_players):
            # if not active hand strength set
            # to 1 worse than worst possible rank
            hand_strength = worst_hand
            if self.active[player]:
                hand_strength = self.judge.evaluate(
                    self.hole_cards[player], self.community_cards
                )
            hand_list.append([player, hand_strength, self.pot_commit[player]])
        hands = np.array(hand_list)
        # sort hands by hand strength and pot commits
        hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
        pot = self.pot
        remainder = 0
        # iterate over hand strength and
        # pot commits from smallest to largest
        for idx, (_, strength, pot_commit) in enumerate(hands):
            eligible = hands[:, 0][hands[:, 1] == strength].astype(int)
            # cut can only be as large as lowest player commit amount
            cut = np.clip(hands[:, 2], None, pot_commit)
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

    def __move_action(self):
        for idx in range(1, self.n_players + 1):
            player = (self.player + idx) % self.n_players
            if self.active[player]:
                break
            else:
                self.acted[player] = True
        self.player = player

    def __clean(self, raise_size):
        if isinstance(raise_size, (int, float)):
            return raise_size
        if raise_size == "pot":
            return raise_size
        raise exceptions.InvalidRaiseSizeError(
            f"unknown raise size, expected one of (int, float, pot),"
            f" got {raise_size}"
        )

    def register_agents(self, agents: Union[List, Dict]) -> None:
        error_msg = "invalid agent configuration, got {}, expected {}"
        if not isinstance(agents, (dict, list)):
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(type(agents), "list or dictionary of agents")
            )
        if len(agents) != self.n_players:
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(
                    f"{len(agents)} number of agents",
                    f"{self.n_players} number of agents",
                )
            )
        if isinstance(agents, list):
            agent_keys = list(range(len(agents)))
        else:
            agent_keys = list(agents.keys())
            if set(agent_keys) != set(range(len(agents))):
                raise exceptions.InvalidAgentConfigurationError(
                    f"invalid agent configuration, got {agent_keys}, "
                    f"expected permutation of {list(range(len(agents)))}"
                )
            agents = list(agents.values())
        all_base_agents = all(isinstance(a, BaseAgent) for a in agents)
        if not all_base_agents:
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(
                    f"agent types {[type(_agent) for _agent in agents]}",
                    "only subtypes of breinforce.agents.BaseAgent",
                )
            )
        self.agents = dict(zip(agent_keys, agents))

    @property
    def state(self):
        output = {
            # auxilary
            "table_name": self.table_name,
            "sb": self.small_blind,
            "bb": self.big_blind,
            "st": self.straddle,
            "hand_name": self.hand_name,
            "date1": self.date1,
            "date2": self.date2,
            "player_names": self.player_names,
            "hole_cards": self.hole_cards,
            "start_stacks": self.start_stacks,
            # config
            "player": self.player,
            "active": self.active,
            "allin": self.active * (self.stacks == 0),
            "community_cards": self.community_cards,
            "button": self.button,
            "done": all(self.__done()),
            "pot": self.pot,
            "payouts": self.payouts,
            "street_commits": self.street_commits,
            "n_players": self.n_players,
            "n_hole_cards": self.n_hole_cards,
            "n_community_cards": sum(self.n_community_cards),
            "rake": self.rake,
            "antes": self.antes,
            "street": self.street,
            "min_raise": self.min_raise,
            "max_raise": self.max_raise,
            "acted": self.acted.copy(),
            "active": self.active.copy(),
            "stacks": self.stacks.copy()
        }
        return output

    def render(self, mode="ascii"):
        return "ascii"
