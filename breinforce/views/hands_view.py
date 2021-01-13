import os
import uuid
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
        screen = self.env.screen()
        hand_id = screen['hand_id']
        sb = screen['sb']
        bb = screen['bb']
        st = screen['st']
        table_id = screen['table_id']
        date1 = screen['date1']
        date2 = screen['date2']
        n_players = screen['n_players']
        button = screen['button']
        player_ids = screen['player_ids']
        stacks = screen['start_stacks']
        hole_cards = screen['hole_cards']
        pot = screen['pot']
        rake = screen['rake']
        community_cards = screen['community_cards']
        flop_cards = repr(community_cards[:3])
        turn_cards = repr(community_cards[:3]) + '[' + repr(community_cards[3]) + ']'
        river_cards = repr(community_cards[:4]) + '[' + repr(community_cards[4]) + ']'

        # Header
        output += f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'(${sb}/${bb}/${st} EUR) - {date1} MSK\n'# [{date2} ET]\n'
        # Table
        output += f'Table \'{table_id}\' {n_players}-max' \
            f'Seat #{button + 1} is the button\n'
        # Seats
        for i, stack in enumerate(stacks):
            user_id = player_ids[i]
            output += f'Seat {i+1}: {user_id} (${stack} in chips)\n'
        # Preflop
        output += self.__subhistory(self.env.history, 0)
        # Dealt
        output += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_id = player_ids[player]
            player_cards = repr(hole_cards[player])
            output += f'Dealt to {player_id} {player_cards}\n'
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
        #print(type(self.env.history))
        # Summary
        output += f'*** SUMMARY ***\n'
        output += f'Total pot {pot} | rake {rake}'
        self.string = output
        return self

    def __str__(self):
        '''String representation of the view
        '''
        return self.string

    def _uuid(self, size, mode='hex'):
        string = ''
        if mode == 'int':
            string = str(uuid.uuid4().int)[:size]
        elif mode == 'hex':
            string = uuid.uuid4().hex[:size]
        return string

    def __subhistory(self, subhistory, street):
        output = ''
        for item in subhistory:
            player, action, info = item
            if info['street'] == street:
                output += f'Player{player+1} '
                if info['action_type'] == 'small_blind':
                    output += f'posts small blind ${action}'
                elif info['action_type'] == 'big_blind':
                    output += f'posts big blind ${action}'
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
