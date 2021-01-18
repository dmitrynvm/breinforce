import os
from .base_view import BaseView


class HandsView(BaseView):
    '''Poker hands history view in the PokerStars format with delayed rendering.
    '''

    def __init__(self, env) -> None:
        self.env = env
        self.string = ''

    def render(self) -> str:
        '''Render representation based on the table configuration
        '''
        out = ''
        state = self.env.state
        self.state = state
        hand_id = state['hand_id']
        sb = state['sb']
        bb = state['bb']
        st = state['st']
        table_name = state['table_name']
        date = state['date']
        n_players = state['n_players']
        button = state['button']
        player_ids = state['player_ids']
        start_stacks = state['start_stacks']
        pot = state['pot']
        rake = state['rake']
        hole_cards = state['hole_cards']
        board_cards = state['board_cards']
        flop_cards = repr(board_cards[:3])
        turn_cards = repr(board_cards[:3]) + '[' + repr(board_cards[3]) + ']'
        river_cards = repr(board_cards[:4]) + '[' + repr(board_cards[4]) + ']'
        self.payouts = state['payouts']
        self.summary = self.__summary(state)

        # Header
        out += f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'(${sb}/${bb}/${st} chips) - {date} MSK\n'# [{date2} ET]\n'
        # Table
        out += f'Table \'{table_name}\' {n_players}-max ' \
            f'Seat #{button + 1} is the button\n'
        # Seats
        for i, start_stack in enumerate(start_stacks):
            player_id = player_ids[i]
            out += f'Seat {i+1}: {player_id} (${start_stack} in chips)\n'
        # Preflop
        #out += self.__subhistory(self.env.history, 0)
        sb_id = player_ids[button + 1]
        out += f'{sb_id} posts small blind ${sb}\n'
        bb_id = player_ids[button + 2]
        out += f'{bb_id} posts big blind ${bb}\n'
        if n_players > 3:
            st_id = player_ids[button + 3]
            out += f'{st_id} posts straddle ${st}\n'
        # Dealt
        out += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_id = player_ids[player]
            player_cards = repr(hole_cards[player])
            out += f'Dealt to {player_id} {player_cards}\n'
        # Preflop
        for item in self.env.history:
            state, player, action, info = item
            if state['street'] == 0:
                out += f'Player{player+1} '
                if info['action_type'] == 'fold':
                    out += 'folds'
                    out += f" (chosen {action} from {info['legal_actions']})"
                elif info['action_type'] == 'check':
                    out += 'checks'
                elif info['action_type'] == 'call':
                    out += f"called ${action} chips"
                elif info['action_type'] == 'raise':
                    out += f"raised from ${info['lower']-action} to ${info['upper']}"
                else:
                    out += f'bets {action}'
                out += '\n'

        # out += self.__subhistory(self.env.history, 1)
        # # Flop
        # flop = self.__subhistory(self.env.history, 2)
        # if flop:
        #     out += f'*** FLOP CARDS *** {flop_cards}\n'
        #     out += flop
        # # Turn
        # turn = self.__subhistory(self.env.history, 3)
        # if turn:
        #     out += f'*** TURN CARDS *** {turn_cards}\n'
        #     out += turn
        # # River
        # river = self.__subhistory(self.env.history, 4)
        # if river:
        #     out += f'*** RIVER CARDS *** {river_cards}\n'
        #     out += river
        # # Summary
        # out += f'*** SUMMARY ***\n'
        # out += f'Total pot {pot} | rake {pot * rake} -> {int(pot * rake)}\n'
        # out += f'Board {board_cards}\n'
        # out += self.__summary()

        self.string = out
        return self

    def __str__(self):
        '''String representation of the view
        '''
        return self.string

    def __subhistory(self, subhistory, street):
        out = ''
        for item in subhistory:
            state, player, action, info = item
            if state['street'] == street:
                out += f'Player{player+1} '
                if info['action_type'] == 'small_blind':
                    out += f'posts small blind ${action}'
                elif info['action_type'] == 'big_blind':
                    out += f'posts big blind ${action}'
                elif info['action_type'] == 'straddle':
                    out += f'posts straddle ${action}'
                elif info['action_type'] == 'fold':
                    out += 'folds'
                elif info['action_type'] == 'check':
                    out += 'checks'
                elif info['action_type'] == 'call':
                    out += f"called ${action} chips"
                elif info['action_type'] == 'raise':
                    out += f"raised from ${info['min_raise']-action} to ${info['min_raise']}"
                else:
                    out += f'bet {action}'
                out += '\n'
        return out

    def __summary(self, state):
        out = ''
        n_players = state['n_players']
        results = self.__results(state)
        # won
        for player, item in enumerate(results):
            out += f"Seat {player+1}: {results[player]['name']} "
            # role
            if results[player]['role']:
                out += f"({results[player]['role']}) "
            # show
            out += f"showed {results[player]['hole']} "
            if results[player]['won']:
                out += f"and won {results[player]['won']} "
                out += f"with {results[player]['rank']}"
            out += '\n'
        return out

    def __results(self, state):
        items = []
        n_players = self.state['n_players']
        payouts = self.payouts
        hole_cards = self.state['hole_cards']
        comm_cards = self.state['board_cards']
        rake = self.state['rake']
        for player in range(n_players):
            item = {
                'name': None,
                'role': None,
                'won': None,
                'hole': hole_cards[player],
                'rank': None,
                'flop': None
            }
            item['name'] = self.state['player_ids'][player]
            if player == 0:
                item['role'] = 'button'
            elif player == 1:
                item['role'] = 'small blind'
            elif player == 2:
                item['role'] = 'big blind'
            elif player == 3:
                item['role'] = 'straddle'
            if payouts[player] > 0:
                item['won'] = (1 - rake) * payouts[player]
            rank_val = self.env.judge.evaluate(hole_cards[player], comm_cards)
            rank = self.env.judge.get_rank_class(rank_val)
            item['rank'] = rank
            items.append(item)
        return items
