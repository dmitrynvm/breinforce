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
        table_id = screen['table_id']
        date1 = screen['date1']
        date2 = screen['date2']
        n_players = screen['n_players']
        button = screen['button']
        player_ids = screen['player_ids']
        stacks = screen['start_stacks']
        hole_cards = screen['hole_cards']
        community_cards = screen['community_cards']
        flop_cards = repr(community_cards[:3])
        turn_cards = repr(community_cards[:3]) + '[' + repr(community_cards[3]) + ']'
        river_cards = repr(community_cards[:4]) + '[' + repr(community_cards[4]) + ']'

        # Header
        output += f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'($sb/$bb EUR) - {date1} MSK [{date2} ET]\n'
        # Table
        output += f'Table \'{table_id}\' {n_players}-max' \
            f'Seat #{button} is the button\n'
        # Seats
        for i, stack in enumerate(stacks):
            user_id = player_ids[i]
            output += f'Seat {i+1}: {user_id} (${stack} in chips)\n'
        # Dealt
        output += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_id = player_ids[player]
            player_cards = repr(hole_cards[player])
            output += f'Dealt to {player_id} {player_cards}\n'
        # Preflop
        output += self.__subhistory(self.env.history, 0)
        # Flop
        output = f'*** FLOP CARDS *** {flop_cards}\n'
        output += self.__subhistory(self.env.history, 1)
        # Preflop
        output = f'*** TURN CARDS *** {turn_cards}\n'
        output += self.__subhistory(self.env.history, 2)
        # River
        output = f'*** RIVER CARDS *** {river_cards}\n'
        output += self.__subhistory(self.env.history, 3)
        #print(type(self.env.history))
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
                if info['folded']:
                    output += 'folds'
                if info['checked']:
                    output += 'checks'
                elif info['called']:
                    output += f"called ${info['called_amount']} chips"
                elif info['raised']:
                    output += f"raised from ${info['raised_to']} to ${info['raised_from']}"
                else:
                    output += f'bet {action}'
                output += '\n'
        return output
