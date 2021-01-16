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
        output = ''
        state = self.env.state
        self.state = state
        hand_name = state['hand_name']
        sb = state['sb']
        bb = state['bb']
        st = state['st']
        table_name = state['table_name']
        date1 = state['date1']
        date2 = state['date2']
        n_players = state['n_players']
        button = state['button']
        player_names = state['player_names']
        start_stacks = state['start_stacks']
        pot = state['pot']
        rake = state['rake']
        hole_cards = state['hole_cards']
        community_cards = state['community_cards']
        flop_cards = repr(community_cards[:3])
        turn_cards = repr(community_cards[:3]) + '[' + repr(community_cards[3]) + ']'
        river_cards = repr(community_cards[:4]) + '[' + repr(community_cards[4]) + ']'
        self.payouts = state['payouts']
        self.summary = self.__summary()

        # Header
        output += f'PokerStars Hand #{hand_name}: Hold\'em No Limit' \
            f'(${sb}/${bb}/${st} chips) - {date1} MSK\n'# [{date2} ET]\n'
        # Table
        output += f'Table \'{table_name}\' {n_players}-max ' \
            f'Seat #{button + 1} is the button\n'
        # Seats
        for i, start_stack in enumerate(start_stacks):
            player_name = player_names[i]
            output += f'Seat {i+1}: {player_name} (${start_stack} in chips)\n'
        # Preflop
        #output += self.__subhistory(self.env.history, 0)
        sb_name = player_names[button + 1]
        output += f'{sb_name} posts small blind ${sb}\n'
        bb_name = player_names[button + 2]
        output += f'{bb_name} posts big blind ${bb}\n'
        if n_players > 3:
            st_name = player_names[button + 3]
            output += f'{st_name} posts straddle ${st}\n'
        # Dealt
        output += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_name = player_names[player]
            player_cards = repr(hole_cards[player])
            output += f'Dealt to {player_name} {player_cards}\n'
        # Preflop
        output += self.__subhistory(self.env.history, 1)
        # Flop
        flop = self.__subhistory(self.env.history, 2)
        if flop:
            output += f'*** FLOP CARDS *** {flop_cards}\n'
            output += flop
        # Turn
        turn = self.__subhistory(self.env.history, 3)
        if turn:
            output += f'*** TURN CARDS *** {turn_cards}\n'
            output += turn
        # River
        river = self.__subhistory(self.env.history, 4)
        if river:
            output += f'*** RIVER CARDS *** {river_cards}\n'
            output += river
        # Summary
        output += f'*** SUMMARY ***\n'
        output += f'Total pot {pot} | rake {pot * rake} -> {int(pot * rake)}\n'
        output += f'Board {community_cards}\n'
        output += self.__summary()

        self.string = output
        return self

    def __str__(self):
        '''String representation of the view
        '''
        return self.string

    def __subhistory(self, subhistory, street):
        output = ''
        for item in subhistory:
            state, player, action, info = item
            if state['street'] == street:
                output += f'Player{player+1} '
                if info['action_type'] == 'small_blind':
                    output += f'posts small blind ${action}'
                elif info['action_type'] == 'big_blind':
                    output += f'posts big blind ${action}'
                elif info['action_type'] == 'straddle':
                    output += f'posts straddle ${action}'
                elif info['action_type'] == 'fold':
                    output += 'folds'
                elif info['action_type'] == 'check':
                    output += 'checks'
                elif info['action_type'] == 'call':
                    output += f"called ${action} chips"
                elif info['action_type'] == 'raise':
                    output += f"raised from ${info['min_raise']-action} to ${info['min_raise']}"
                else:
                    output += f'bet {action}'
                output += '\n'
        return output

    def __summary(self):
        output = ''
        n_players = self.state['n_players']
        results = self.__results()
        # won
        for player, item in enumerate(results):
            output += f"Seat {player+1}: {results[player]['name']} "
            # role
            if results[player]['role']:
                output += f"({results[player]['role']}) "
            # show
            output += f"showed {results[player]['hole']} "
            if results[player]['won']:
                output += f"and won {results[player]['won']} "
                output += f"with {results[player]['rank']}"
            output += '\n'
        return output

    def __results(self):
        items = []
        n_players = self.state['n_players']
        payouts = self.payouts
        hole_cards = self.state['hole_cards']
        comm_cards = self.state['community_cards']
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
            item['name'] = self.state['player_names'][player]
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
