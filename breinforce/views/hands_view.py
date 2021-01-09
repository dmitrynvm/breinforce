import os
import uuid
from datetime import datetime
# from zoneinfo import ZoneInfo , timezone
from . import BaseView


class HandsView(BaseView):
    '''Poker hands history view in the PokerStars format with delayed rendering.
    '''

    def __init__(self, config) -> None:
        self.config = config
        self.string = ''

    def render(self, history) -> str:
        '''Render representation based on the table configuration
        '''
        hand_id = 'h' + self._uuid(12, 'int')
        date1 = datetime.now().strftime('%b/%d/%Y %H:%M:%S')
        date2 = datetime.now().strftime('%b/%d/%Y %H:%M:%S')
        table_id = 't' + self._uuid(4, 'hex')
        num_players = self.config['num_players']
        button = self.config['button']
        user_ids = ['p' + self._uuid(4, 'hex') for _ in range(num_players)]
        stacks = self.config['stacks']
        header = f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'($sb/$bb EUR) - {date1} MSK [{date2} ET]\n'
        preflop = f'Table \'{table_id}\' {num_players}-max' \
            f'Seat #{button} is the button\n'
        for i, stack in enumerate(stacks):
            user_id = user_ids[i]
            preflop += f'Seat {i}: {user_id} (${stack} in chips)\n'
        self.string = header + preflop
        for item in history:
            print(item[0])
        return self

    def __str__(self):
        '''Returns string representation of the view
        '''
        return self.string

    def _uuid(self, size, mode='hex'):
        string = ''
        if mode == 'int':
            string = str(uuid.uuid4().int)[:size]
        elif mode == 'hex':
            string = uuid.uuid4().hex[:size]
        return string
