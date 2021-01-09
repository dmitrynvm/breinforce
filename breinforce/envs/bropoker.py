from typing import Dict, List, Optional, Tuple, Union
import gym
import numpy as np
from breinforce import errors
from breinforce.agents import BaseAgent
from breinforce.games.breinforce import Dealer


class Bropoker(gym.Env):
    '''Runs a range of different of poker games dependent on the
    given configuration. Supports limit, no limit and pot limit
    bet sizing, arbitrary deck sizes, arbitrary hole and community
    cards and many other options.

    Parameters
    ----------
    num_players : int
        maximum number of players
    num_streets : int
        number of streets including preflop, e.g. for texas hold'em
        num_streets=4
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
    num_raises : Union[float, List[float]]
        max number of bets for each street including preflop, valid
        raise numbers are ints and floats. if only a single int or float
        is passed the value is expanded to a list the length of number
        of streets, e.g. for a standard limit game pass num_raises=4
    num_suits : int
        number of suits to use in deck, must be between 1 and 4
    num_ranks : int
        number of ranks to use in deck, must be between 1 and 13
    num_hole_cards : int
        number of hole cards per player, must be greater than 0
    num_community_cards : Union[int, List[int]]
        number of community cards per street including preflop, e.g.
        for texas hold'em pass num_community_cards=[0, 3, 1, 1]. if only
        a single int is passed, it is expanded to a list the length of
        number of streets
    num_cards_for_hand : int
        number of cards for a valid poker hand, e.g. for texas hold'em
        num_cards_for_hand=5
    start_stack : int
        number of chips each player starts with
    low_end_straight : bool, optional
        toggle to include the low ace straight within valid hands, by
        default True
    order : Optional[List[str]], optional
        optional custom order of hand ranks, must be permutation of
        ['sf', 'fk', 'fh', 'fl', 'st', 'tk', 'tp', 'pa', 'hc']. if
        order=None, hands are ranked by rarity. by default None

    Examples
    ----------
    1-2 Heads Up No Limit Texas Hold'em:

        Dealer(num_players=2, num_streets=4, blinds=[1, 2], antes=0,
               raise_sizes=float('inf'), num_raises=float('inf'),
               num_suits=4, num_ranks=13, num_hole_cards=2, start_stack=200)

    1-2 6 Player PLO

        Dealer(num_players=6, num_streets=4, blinds=[0, 1, 2, 0, 0, 0],
               antes=0, raise_sizes='pot', num_raises=float('inf'),
               num_suits=4, num_ranks=13, num_hole_cards=4, start_stack=200)

    1-2 Heads Up No Limit Short Deck

        Dealer(num_players=2, num_streets=4, blinds=[1, 2], antes=0,
               raise_sizes=float('inf'), num_raises=float('inf'),
               num_suits=4, num_ranks=9, num_hole_cards=2, start_stack=200,
               order=['sf', 'fk', 'fl', 'fh', 'st',
                      'tk', 'tp', 'pa', 'hc'])
    '''

    def __init__(
        self,
        num_players: int,
        num_streets: int,
        blinds: Union[int, List[int]],
        antes: Union[int, List[int]],
        raise_sizes: Union[float, str, List[Union[float, str]]],
        num_raises: Union[float, List[float]],
        num_suits: int,
        num_ranks: int,
        num_hole_cards: int,
        num_community_cards: Union[int, List[int]],
        num_cards_for_hand: int,
        start_stack: int,
        low_end_straight: bool = True,
        rake: float = 0.0,
        order: Optional[List[str]] = None,
    ) -> None:

        self.dealer = Dealer(
            num_players,
            num_streets,
            blinds,
            antes,
            raise_sizes,
            num_raises,
            num_suits,
            num_ranks,
            num_hole_cards,
            num_community_cards,
            num_cards_for_hand,
            start_stack,
            low_end_straight,
            rake,
            order
        )

        max_bet = start_stack * num_players
        num_comm_cards = sum(num_community_cards)

        self.action_space = gym.spaces.Discrete(max_bet)
        self.observation_space = gym.spaces.Dict(
            {
                'player': gym.spaces.Discrete(num_players),
                'active': gym.spaces.MultiBinary(num_players),
                'button': gym.spaces.Discrete(num_players),
                'call': gym.spaces.Discrete(max_bet),
                'community_cards': gym.spaces.Tuple(
                    (
                        gym.spaces.Tuple(
                            (
                                gym.spaces.Discrete(num_ranks),
                                gym.spaces.Discrete(num_suits)
                            )
                        ),
                    ) * num_comm_cards
                ),
                'hole_cards': gym.spaces.Tuple(
                    (
                        gym.spaces.Tuple(
                            (
                                gym.spaces.Tuple(
                                    (
                                        gym.spaces.Discrete(num_ranks),
                                        gym.spaces.Discrete(num_suits)
                                    )
                                ),
                            ) * num_hole_cards
                        ),
                    ) * num_players
                ),
                'min_raise': gym.spaces.Discrete(max_bet),
                'max_raise': gym.spaces.Discrete(max_bet),
                'pot': gym.spaces.Discrete(max_bet),
                'stacks': gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(max_bet),
                    ) * num_players
                ),
                'street_commits': gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(max_bet),
                    ) * num_players
                )
            }
        )

        self.agents: Optional[Dict[int, BaseAgent]] = None
        self.prev_obs: Optional[Dict] = None

    def act(self, obs: dict) -> int:
        if self.agents is None:
            raise errors.NoRegisteredAgentsError(
                'register agents using env.register_agents(...) before'
                'calling act(obs)'
            )
        if self.prev_obs is None:
            raise errors.EnvironmentResetError(
                'call reset() before calling first step()'
            )
        player = self.prev_obs['player']
        bet = self.agents[player].act(obs)
        return bet

    def step(self, bet: int) -> Tuple[Dict, np.ndarray, np.ndarray, None]:
        obs, rewards, done = self.dealer.step(bet)
        obs['hole_cards'] = obs['hole_cards'][obs['player']]
        if self.agents is not None:
            self.prev_obs = obs
        return obs, rewards, done, None

    def reset(
        self,
        reset_button: bool = False,
        reset_stacks: bool = False
    ) -> Dict:
        obs = self.dealer.reset(reset_button, reset_stacks)
        obs['hole_cards'] = obs['hole_cards'][obs['player']]
        if self.agents is not None:
            self.prev_obs = obs
        return obs

    def register_agents(self, agents: Union[List, Dict]) -> None:
        error_msg = 'invalid agent configuration, got {}, expected {}'
        if not isinstance(agents, (dict, list)):
            raise errors.InvalidAgentConfigurationError(
                error_msg.format(type(agents), 'list or dictionary of agents')
            )
        if len(agents) != self.dealer.num_players:
            raise errors.InvalidAgentConfigurationError(
                error_msg.format(
                    f'{len(agents)} number of agents',
                    f'{self.dealer.num_players} number of agents',
                )
            )
        if isinstance(agents, list):
            agent_keys = list(range(len(agents)))
        else:
            agent_keys = list(agents.keys())
            if set(agent_keys) != set(range(len(agents))):
                raise errors.InvalidAgentConfigurationError(
                    f'invalid agent configuration, got {agent_keys}, '
                    f'expected permutation of {list(range(len(agents)))}'
                )
            agents = list(agents.values())
        all_base_agents = all(isinstance(a, BaseAgent) for a in agents)
        if not all_base_agents:
            raise errors.InvalidAgentConfigurationError(
                error_msg.format(
                    f'agent types {[type(_agent) for _agent in agents]}',
                    'only subtypes of breinforce.agents.BaseAgent',
                )
            )
        self.agents = dict(zip(agent_keys, agents))
