'''Classes and functions for running poker games'''
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
# from zoneinfo import ZoneInfo , timezone


class Bropoker(gym.Env):
    '''Runs a range of different of poker games dependent on the
    given configuration. Supports limit, no limit and pot limit
    bet sizing, arbitrary deck sizes, arbitrary hole and community
    cards and many other options.

    Parameters
    ----------
    n_players : int
        maximum number of players
    n_streets : int
        number of streets including preflop, e.g. for texas hold'em
        n_streets=4
    blinds : Union[int, List[int]]
        blind distribution as a list of ints, one for each player
        starting from the button e.g. [0, 1, 2] for a three player game
        with a sb of 1 and bb of 2, passed ints will be expanded to
        all players i.e. pass blinds=0 for no blinds
    antes : Union[int, List[int]]
        ante distribution as a list of ints, one for each player
        starting from the button e.g. [0, 0, 5] for a three player game
        with a bb ante of 5, passed ints will be expanded to all
        players i.e. pass antes=0 for no antes
    raise_sizes : Union[float, str, List[Union[float, str]]]
        max raise sizes for each street, valid raise sizes are ints,
        floats, and 'pot', e.g. for a 1-2 limit hold'em the raise sizes
        should be [2, 2, 4, 4] as the small and big bet are 2 and 4.
        float('inf') can be used for no limit games. pot limit raise
        sizes can be set using 'pot'. if only a single int, float or
        string is passed the value is expanded to a list the length
        of number of streets, e.g. for a standard no limit game pass
        raise_sizes=float('inf')
    n_raises : Union[float, List[float]]
        max number of bets for each street including preflop, valid
        raise numbers are ints and floats. if only a single int or float
        is passed the value is expanded to a list the length of number
        of streets, e.g. for a standard limit game pass n_raises=4
    n_suits : int
        number of suits to use in deck, must be between 1 and 4
    n_ranks : int
        number of ranks to use in deck, must be between 1 and 13
    n_hole_cards : int
        number of hole cards per player, must be greater than 0
    n_community_cards : Union[int, List[int]]
        number of community cards per street including preflop, e.g.
        for texas hold'em pass n_community_cards=[0, 3, 1, 1]. if only
        a single int is passed, it is expanded to a list the length of
        number of streets
    n_cards_for_hand : int
        number of cards for a valid poker hand, e.g. for texas hold'em
        n_cards_for_hand=5
    start_stack : int
        number of chips each player starts with
    low_end_straight : bool, optional
        toggle to include the low ace straight within valid hands, by
        default True
    order : Optional[List[str]], optional
        optional custom order of hand ranks, must be permutation of
        ['sf', 'fk', 'fh', 'fl', 'st', 'tk', 'tp', 'pa', 'hc']. if
        order=None, hands are ranked by rarity. by default None
    '''

    def __init__(
        self,
        n_players: int,
        n_streets: int,
        blinds: Union[int, List[int]],
        antes: Union[int, List[int]],
        raise_sizes: Union[float, str, List[Union[float, str]]],
        n_raises: Union[float, List[float]],
        n_suits: int,
        n_ranks: int,
        n_hole_cards: int,
        n_community_cards: Union[int, List[int]],
        n_cards_for_hand: int,
        start_stack: int,
        low_end_straight: bool = True,
        rake: float = 0.0,
        order: Optional[List[str]] = None,
    ) -> None:

        # config
        self.n_players = n_players
        self.n_streets = n_streets
        self.blinds = np.array(blinds)
        self.antes = np.array(antes)
        self.big_blind = blinds[1]
        self.raise_sizes = [self.__clean_rs(rs) for rs in raise_sizes]
        self.n_raises = [float(raise_num) for raise_num in n_raises]
        self.n_suits = n_suits
        self.n_ranks = n_ranks
        self.n_hole_cards = n_hole_cards
        self.n_community_cards = n_community_cards
        self.n_cards_for_hand = n_cards_for_hand
        self.start_stack = start_stack

        # auxilary
        self.hand_id = 'h' + self._uuid(12, 'int')
        self.date1 = datetime.now().strftime('%b/%d/%Y %H:%M:%S')
        self.date2 = datetime.now().strftime('%b/%d/%Y %H:%M:%S')
        self.table_id = 'table1'# + self._uuid(5, 'hex')
        self.player_ids = [
            'agent' + str(i+1) for i in range(self.n_players)
        ]
        self.start_stacks = np.full(
            self.n_players,
            self.start_stack,
            dtype=np.int32
        )

        # dealer
        self.player = -1
        self.active = np.zeros(self.n_players, dtype=np.uint8)
        self.button = 0
        self.community_cards: List[Card] = []
        self.deck = Deck(self.n_suits, self.n_ranks)
        self.rake = rake
        self.judge = Judge(
            self.n_suits,
            self.n_ranks,
            self.n_cards_for_hand,
            low_end_straight=low_end_straight,
            order=order,
        )
        self.history = []
        self.hole_cards: List[List[Card]] = []
        self.largest_raise = 0
        self.pot = 0
        self.pot_commit = np.zeros(self.n_players, dtype=np.int32)
        self.stacks = np.full(
            self.n_players,
            self.start_stack,
            dtype=np.int32
        )
        self.street = 0
        self.street_commits = np.zeros(self.n_players, dtype=np.int32)
        self.street_option = np.zeros(self.n_players, dtype=np.uint8)
        self.street_raises = 0

        self.config = {
            'n_players': self.n_players,
            'n_streets': self.n_streets,
            'blinds': self.blinds,
            'antes': self.antes,
            'raise_sizes': self.raise_sizes,
            'n_raises': self.n_raises,
            'n_suits': self.n_suits,
            'n_ranks': self.n_ranks,
            'n_hole_cards': self.n_hole_cards,
            'n_community_cards': self.n_community_cards,
            'n_cards_for_hand': self.n_cards_for_hand,
            'start_stack': self.start_stack,
            'rake': self.rake,
            'button': self.button + 1,
            'stacks': self.stacks
        }

        self.agents: Optional[Dict[int, BaseAgent]] = None
        self.prev_obs: Optional[Dict] = None

    @staticmethod
    def configure():
        """
        Merges the local configured envs to the global OpenAI Gym list.
        """
        try:
            config_path = os.path.join(CONFIG_DIR, 'bropoker.json')
            with open(config_path, 'r') as f:
                configs = json.loads(f.read())
                for name, config in configs.items():
                    configs[name] = utils.parse_config(config)
                    Bropoker.register(configs)
        except ImportError:
            pass

    @staticmethod
    def register(configs: Dict) -> None:
        '''Registers dict of breinforce configs as gym environments

        Parameters
        ----------
        configs : Dict
            dictionary of bropoker configs, keys must environment ids and
            values valid bropoker configs, example:
                configs = {
                    'NolimitHoldemTwoPlayer-v0': {
                        'n_players': 2,
                        'n_streets': 4,
                        'blinds': [1, 2],
                        'antes': 0,
                        'raise_sizes': float('inf'),
                        'n_raises': float('inf'),
                        'n_suits': 4,
                        'n_ranks': 13,
                        'n_hole_cards': 2,
                        'n_community_cards': [0, 3, 1, 1],
                        'n_cards_for_hand': 5,
                        'start_stack': 200
                    }
                }
        '''
        env_entry_point = 'breinforce.envs:Bropoker'
        env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]
        for env_id, config in configs.items():
            if env_id not in env_ids:
                gym.envs.registration.register(
                    id=env_id, entry_point=env_entry_point, kwargs={**config}
                )

    def reset(self) -> Dict:
        '''Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.

        Parameters
        ----------
        reset_button : bool, optional
            reset button to first position at table, by default False
        reset_stacks : bool, optional
            reset stack sizes to starting stack size, by default False

        Returns
        -------
        Dict
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
        '''
        self.active.fill(1)
        self.stacks = np.full(self.n_players, self.start_stack)
        self.button = 0
        self.deck.shuffle()
        self.community_cards = self.deck.draw(self.n_community_cards[0])
        self.history = []#{street: [] for street in range(self.n_streets)}
        self.hole_cards = [
            self.deck.draw(self.n_hole_cards) for _
            in range(self.n_players)
        ]
        self.largest_raise = self.big_blind
        self.pot = 0
        self.pot_commit.fill(0)
        self.street = 0
        self.street_commits.fill(0)
        self.street_option.fill(0)
        self.street_raises = 0

        self.player = self.button
        # in heads up button posts small blind
        if self.n_players > 2:
            self.__move_action()
        self.__collect_multiple_actions(actions=self.antes, street_commits=False)
        self.__collect_multiple_actions(actions=self.blinds, street_commits=True)
        self.__move_action()
        self.__move_action()
        return self.__observation()

    def act(self, obs: dict) -> int:
        if self.agents is None:
            raise exceptions.NoRegisteredAgentsError(
                'register agents using env.register_agents(...) before'
                'calling act(obs)'
            )
        if self.prev_obs is None:
            raise exceptions.EnvironmentResetError(
                'call reset() before calling first step()'
            )
        player = self.prev_obs['player']
        action = self.agents[player].act(obs)
        return action

    def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        '''Advances poker game to next player. If the action is 0, it is
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
        '''
        if self.player == -1:
            if any(self.active):
                return self.__output()
            raise exceptions.HashMapResetError(
                'call reset() before calling first step()'
            )

        fold = action < 0
        action = round(action)
        folded = fold

        call, min_raise, max_raise = self.__action_sizes()
        # round action to nearest sizing
        action = self.__clean_action(action, call, min_raise, max_raise)

        # only fold if player cannot check
        #
        called = call > 0 and action == call
        checked = call == 0 and action == 0

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

        self.__collect_action(action)
        action = int(action)
        info = {
            'street': self.street,
            'folded': folded,
            'checked': checked,
            'called': called,
            'called_amount': call,
            'raised': raised,
            'raised_from': call,
            'raised_to': call + action
        }
        self.history.append((self.player, action, info))
        self.street_option[self.player] = True
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
            self.street_option = np.logical_not(self.active).astype(np.uint8)
            self.street_raises = 0

        obs, payouts, done, info = self.__output()
        if all(done):
            self.player = -1
            obs['player'] = -1
            if self.rake != 0.0:
                payouts = [
                    int(p * (1 - self.rake)) if p > 0 else p for p in payouts
                ]
        obs['hole_cards'] = obs['hole_cards'][obs['player']]
        return obs, payouts, done, None

    def __all_agreed(self) -> bool:
        # not all agreed if not all players had chance to act
        if not all(self.street_option):
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
            min_raise = max(self.big_blind, self.largest_raise + call)
            if self.raise_sizes[self.street] == 'pot':
                max_raise = self.pot + call * 2
            elif self.raise_sizes[self.street] == float('inf'):
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

    def __collect_multiple_actions(
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

    def __collect_action(self, action: int):
        # action only as large as stack size
        action = min(self.stacks[self.player], action)

        self.pot += action
        self.pot_commit[self.player] += action
        self.street_commits[self.player] += action
        self.stacks[self.player] -= action

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
            'player': self.player,
            'active': self.active,
            'button': self.button,
            'call': call,
            'community_cards': community_cards,
            'hole_cards': hole_cards,
            'max_raise': max_raise,
            'min_raise': min_raise,
            'pot': self.pot,
            'stacks': self.stacks,
            'street_commits': self.street_commits,
        }
        if self.agents is not None:
            self.prev_obs = obs
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
                self.street_option[player] = True
        self.player = player

    def __clean_rs(self, raise_size):
        if isinstance(raise_size, (int, float)):
            return raise_size
        if raise_size == 'pot':
            return raise_size
        raise exceptions.InvalidRaiseSizeError(
            f'unknown raise size, expected one of (int, float, pot),'
            f' got {raise_size}'
        )

    def register_agents(self, agents: Union[List, Dict]) -> None:
        error_msg = 'invalid agent configuration, got {}, expected {}'
        if not isinstance(agents, (dict, list)):
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(type(agents), 'list or dictionary of agents')
            )
        if len(agents) != self.n_players:
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(
                    f'{len(agents)} number of agents',
                    f'{self.n_players} number of agents',
                )
            )
        if isinstance(agents, list):
            agent_keys = list(range(len(agents)))
        else:
            agent_keys = list(agents.keys())
            if set(agent_keys) != set(range(len(agents))):
                raise exceptions.InvalidAgentConfigurationError(
                    f'invalid agent configuration, got {agent_keys}, '
                    f'expected permutation of {list(range(len(agents)))}'
                )
            agents = list(agents.values())
        all_base_agents = all(isinstance(a, BaseAgent) for a in agents)
        if not all_base_agents:
            raise exceptions.InvalidAgentConfigurationError(
                error_msg.format(
                    f'agent types {[type(_agent) for _agent in agents]}',
                    'only subtypes of breinforce.agents.BaseAgent',
                )
            )
        self.agents = dict(zip(agent_keys, agents))

    def _uuid(self, size, mode='hex'):
        string = ''
        if mode == 'int':
            string = str(uuid.uuid4().int)[:size]
        elif mode == 'hex':
            string = uuid.uuid4().hex[:size]
        return string

    def screen(self):
        output = {
            # auxilary
            'table_id': self.table_id,
            'hand_id': self.hand_id,
            'date1': self.date1,
            'date2': self.date2,
            'player_ids': self.player_ids,
            'hole_cards': self.hole_cards,
            'start_stacks': self.start_stacks,
            # config
            'player': self.player,
            'active': self.active,
            'allin': self.active * (self.stacks == 0),
            'community_cards': self.community_cards,
            'button': self.button,
            'done': all(self.__done()),
            'hole_cards': self.hole_cards,
            'pot': self.pot,
            'payouts': self.__payouts(),
            'prev_action': None if not self.history else self.history[-1],
            'street_commits': self.street_commits,
            'stacks': self.stacks,
            'n_players': self.n_players,
            'n_hole_cards': self.n_hole_cards,
            'n_community_cards': sum(self.n_community_cards)
        }
        return output
