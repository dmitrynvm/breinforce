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
        state = self.env.state()
        self.state = state
        hand_id = state['hand_id']
        sb = state['sb']
        bb = state['bb']
        st = state['st']
        table_id = state['table_id']
        date1 = state['date1']
        date2 = state['date2']
        n_players = state['n_players']
        button = state['button']
        player_ids = state['player_ids']
        stacks = state['start_stacks']
        hole_cards = state['hole_cards']
        pot = state['pot']
        rake = state['rake']
        community_cards = state['community_cards']
        flop_cards = repr(community_cards[:3])
        turn_cards = repr(community_cards[:3]) + '[' + repr(community_cards[3]) + ']'
        river_cards = repr(community_cards[:4]) + '[' + repr(community_cards[4]) + ']'
        self.payouts = state['payouts']
        self.summary = self.__summary()

        # Header
        output += f'PokerStars Hand #{hand_id}: Hold\'em No Limit' \
            f'(${sb}/${bb}/${st} chips) - {date1} MSK\n'# [{date2} ET]\n'
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
            state, player, action, info = item
            if info['street'] == street:
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
