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

    def __init__(self, env, tpl_name='ascii_table.txt') -> None:
        self.env = env
        tpl_path = os.path.join(VIEW_DIR, tpl_name)
        f = open(tpl_path, 'r')
        self.template = f.read()

    def render(self) -> str:
        '''Render ascii table representation based on the table
        state
        '''
        state = self.env.state()
        self.n_players = state['n_players']
        self.n_hole_cards = state['n_hole_cards']
        self.n_community_cards = state['n_community_cards']
        self.player_pos = self.POS_DICT[self.n_players]
        player = state['player']
        button = state['button']
        done = state['done']
        positions = ['p{}'.format(idx) for idx in self.player_pos]

        players = self._parse_players(state, done, player)
        action_string, win_string = self._parse_string(state, done, positions)

        str_state = {key: '' for key in self.KEYS}

        # community cards
        ccs = [str(card) for card in state['community_cards']]
        ccs += ['--'] * (self.n_community_cards - len(ccs))
        ccs_string = '[' + ','.join(ccs) + ']'
        str_state['ccs'] = ccs_string

        # pot
        if not done:
            str_state['pot'] = '{:,}'.format(state['pot'])
            str_state['a{}'.format(self.player_pos[player])] = 'X'
        else:
            str_state['pot'] = '0'

        # button + player positions
        str_state['b{}'.format(self.player_pos[button])] = 'B'
        iterables = [
            players,
            state['street_commits'],
            positions,
            state['allin']
        ]
        for player, street_commit, pos, allin in zip(*iterables):
            str_state[pos] = player
            str_state[pos + 'c'] = '{:,}'.format(street_commit)
            if allin and not done:
                str_state['a' + pos[1:]] = 'A'

        # payouts
        if done:
            iterables = [state['payouts'], positions]
            for payout, pos in zip(*iterables):
                str_state[pos + 'c'] = '{:,}'.format(payout)

        # player + win string
        str_state['player'] = action_string
        str_state['win'] = win_string

        string = self.template.format(**str_state)

        return string

    def _parse_players(self, state, done, player):
        players = []
        iterator = zip(
            state['hole_cards'],
            state['stacks'],
            state['active']
        )
        for idx, (hand, stack, active) in enumerate(iterator):
            if active:
                players.append(
                    '{:2}. '.format(idx + 1)
                    + ','.join([str(card) for card in hand])
                    + ' {:,}'.format(stack)
                )
            else:                
                players.append(
                    '{:2}. '.format(idx + 1)
                    + ','.join(['--'] * self.n_hole_cards)
                    + ' {:,}'.format(stack)
                )

        return players

    def _parse_string(self, state, done, positions):
        action_string = ''
        win_string = ''

        prev_action = state['prev_action']
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
            if sum(payout > 0 for payout in state['payouts']) > 1:
                win_string += 's {} won {} respectively'
            else:
                win_string += ' {} won {}'
            players = []
            payouts = []
            for player, payout in enumerate(state['payouts']):
                if payout > 0:
                    players.append(str(player + 1))
                    payouts.append(str(payout))
            win_string = win_string.format(
                ', '.join(players),
                ', '.join(payouts)
            )
        else:
            action_string += 'Action on Player {}'.format(state['player'] + 1)

        return action_string, win_string
