import os
from breinforce.views import BaseView
from breinforce.config.application import VIEW_DIR


class AsciiView(BaseView):
    '''Poker game renderer which prints an ascii representation of the
    table state to the terminal
    '''

    POS_DICT = {
        2: [0, 5],
        3: [0, 3, 6],
        4: [0, 2, 4, 6],
        5: [0, 2, 4, 6, 8],
        6: [0, 1, 3, 5, 6, 8],
        7: [0, 1, 3, 5, 6, 7, 9],
        8: [0, 1, 2, 4, 5, 6, 7, 9],
        9: [0, 1, 2, 4, 5, 6, 7, 8, 9]
    }

    KEYS = (
        ['p{}'.format(idx) for idx in range(10)]
        + ['p{}c'.format(idx) for idx in range(10)]
        + ['a{}'.format(idx) for idx in range(10)]
        + ['b{}'.format(idx) for idx in range(10)]
        + ['sb', 'bb', 'ccs', 'pot', 'player']
    )

    def __init__(self, tpl_name='ascii_table.txt') -> None:
        tpl_path = os.path.join(VIEW_DIR, tpl_name)
        f = open(tpl_path, 'r')
        self.template = f.read()

    def _parse_players(self, config, done, player):
        players = []
        iterator = zip(
            config['hole_cards'],
            config['stacks'],
            config['active']
        )
        for idx, (hand, stack, active) in enumerate(iterator):
            if not active:
                players.append(
                    '{:2}. '.format(idx + 1)
                    + ','.join(['--'] * self.n_hole_cards)
                    + ' {:,}'.format(stack)
                )
                continue
            if done or idx == player:
                players.append(
                    '{:2}. '.format(idx + 1)
                    + ','.join([str(card) for card in hand])
                    + ' {:,}'.format(stack)
                )
                continue
            players.append(
                '{:2}. '.format(idx + 1)
                + ','.join(['??'] * self.n_hole_cards)
                + ' {:,}'.format(stack)
            )
        return players

    def _parse_string(self, config, done, positions):
        action_string = ''
        win_string = ''

        prev_action = config['prev_action']
        if prev_action is not None:
            action_string = 'Player {} {}'
            player, action, fold = prev_action
            if fold:
                action = 'folded '
            else:
                if action:
                    action = 'action {} '.format(action)
                else:
                    action = 'checked '
            action_string = action_string.format(player + 1, action)

        if done:
            win_string = 'Player'
            if sum(payout > 0 for payout in config['payouts']) > 1:
                win_string += 's {} won {} respectively'
            else:
                win_string += ' {} won {}'
            players = []
            payouts = []
            for player, payout in enumerate(config['payouts']):
                if payout > 0:
                    players.append(str(player + 1))
                    payouts.append(str(payout))
            win_string = win_string.format(
                ', '.join(players),
                ', '.join(payouts)
            )
        else:
            action_string += 'Action on Player {}'.format(config['player'] + 1)

        return action_string, win_string

    def render(self, config: dict, **kwargs) -> str:
        '''Render ascii table representation based on the table
        configuration

        Parameters
        ----------
        config : dict
            game configuration dictionary,
                config = {
                    'player': int - position of active player,
                    'active': List[bool] - list of active players,
                    'allin': List[bool] - list of all in players,
                    'community_cards': List[Card] - list of community
                                       cards,
                    'button': int - position of button,
                    'done': bool - list of done players,
                    'hole_cards': List[List[Card]] - list of hole cards,
                    'pot': int - chips in pot,
                    'payouts': List[int] - list of chips won for each
                               player,
                    'prev_action': Tuple[int, int, int] - last
                                   position action and fold,
                    'street_commits': List[int] - list of number of
                                      chips added to pot from each
                                      player on current street,
                    'stacks': List[int] - list of stack sizes,
                }
        '''

        self.n_players = config['n_players']
        self.n_hole_cards = config['n_hole_cards']
        self.n_community_cards = config['n_community_cards']
        self.player_pos = self.POS_DICT[self.n_players]
        player = config['player']
        button = config['button']
        done = config['done']
        positions = ['p{}'.format(idx) for idx in self.player_pos]

        players = self._parse_players(config, done, player)
        action_string, win_string = self._parse_string(config, done, positions)

        str_config = {key: '' for key in self.KEYS}

        # community cards
        ccs = [str(card) for card in config['community_cards']]
        ccs += ['--'] * (self.n_community_cards - len(ccs))
        ccs_string = '[' + ','.join(ccs) + ']'
        str_config['ccs'] = ccs_string

        # pot
        if not done:
            str_config['pot'] = '{:,}'.format(config['pot'])
            str_config['a{}'.format(self.player_pos[player])] = 'X'
        else:
            str_config['pot'] = '0'

        # button + player positions
        str_config['b{}'.format(self.player_pos[button])] = 'D '
        iterables = [
            players,
            config['street_commits'],
            positions,
            config['allin']
        ]
        for player, street_commit, pos, allin in zip(*iterables):
            str_config[pos] = player
            str_config[pos + 'c'] = '{:,}'.format(street_commit)
            if allin and not done:
                str_config['a' + pos[1:]] = 'A'

        # payouts
        if done:
            iterables = [config['payouts'], positions]
            for payout, pos in zip(*iterables):
                str_config[pos + 'c'] = '{:,}'.format(payout)

        # player + win string
        str_config['player'] = action_string
        str_config['win'] = win_string

        string = self.template.format(**str_config)

        return string
