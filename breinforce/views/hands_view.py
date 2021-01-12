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
        screen = self.env.screen()
        hand_id = screen['hand_id']
        table_id = screen['table_id']
        date1 = screen['date1']
        date2 = screen['date2']
        n_players = screen['n_players']
        button = screen['button']
        player_ids = screen['player_ids']
        stacks = screen['stacks']
        hole_cards = screen['hole_cards']
        header = f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'($sb/$bb EUR) - {date1} MSK [{date2} ET]\n'
        preflop = f'Table \'{table_id}\' {n_players}-max' \
            f'Seat #{button} is the button\n'
        for i, stack in enumerate(stacks):
            user_id = player_ids[i]
            preflop += f'Seat {i}: {user_id} (${stack} in chips)\n'
        preflop += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_id = player_ids[player]
            cards = str([repr(card) for card in hole_cards[player]])
            preflop += f'Dealt to {player_id} {cards}\n'
        #for item in history:
        #    print(item[0])
        self.string = header + preflop
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
