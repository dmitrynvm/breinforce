import os


class HandsView():
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
        flop_cards = repr(board_cards[:3]).replace(",", '') if len(board_cards) > 2 else None
        turn_cards = repr(board_cards[:3]).replace(",", '') + '[' + repr(board_cards[3]) + ']' if len(board_cards) > 3 else None
        river_cards = repr(board_cards[:4]).replace(",", '') + '[' + repr(board_cards[4]).replace(",", '') + ']' if len(board_cards) > 4 else None
        self.payouts = state['payouts']

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
        out += f'{sb_id}: posts small blind ${sb}\n'
        bb_id = player_ids[button + 2]
        out += f'{bb_id}: posts big blind ${bb}\n'
        if n_players > 3:
            st_id = player_ids[button + 3]
            out += f'{st_id}: posts straddle ${st}\n'
        # Dealt
        out += '*** HOLE CARDS ***\n'
        for player in range(n_players):
            player_id = player_ids[player]
            player_cards = repr(hole_cards[player]).replace(",", '')
            out += f'Dealt to {player_id} {player_cards}\n'

        preflop = self.select(0)
        out += self.fmt(preflop)

        flop = self.select(1)
        if flop:
            out += f'*** FLOP *** {flop_cards}\n'
            out += self.fmt(flop)

        turn = self.select(2)
        if turn:
            out += f'*** TURN *** {turn_cards}\n'
            out += self.fmt(turn)

        river = self.select(3)
        if river:
            out += f'*** RIVER *** {river_cards}\n'
            out += self.fmt(river)

        # Summary
        out += '*** SUMMARY ***\n'
        out += f'Total pot ${pot} | rake ${int(pot * rake)} \n'
        cards = repr(board_cards).replace(",", '')
        out += f'Board {cards}\n'
        out += self.summary(state)

        self.string = out
        return self.string

    def __str__(self):
        '''String representation of the view
        '''
        return self.string

    def select(self, street):
        out = []
        for item in self.env.history:
            state, player, action, info = item
            if state['street'] == street:
                out.append(item)
        return out

    def summary(self, state):
        out = ''
        n_players = state['n_players']
        results = self.results(state)
        # won
        for player, item in enumerate(results):
            out += f"Seat {player+1}: {results[player]['name']} "
            # role
            fold = results[player]['fold']
            role = results[player]['role']
            if role:
                out += f"({role}) "
            if fold < state['street']:
                out += "folded before Flop"
            elif results[player]['won']:
                out += f"showed {results[player]['hole']} "
                out += f"and won ${results[player]['won']} "
                out += f"with {results[player]['rank']}"
            else:
                out += f"showed {results[player]['hole']} "
            out += '\n'
        return out

    def results(self, state):
        items = []
        n_players = self.state['n_players']
        n_players = self.state['n_players']
        payouts = self.payouts
        hole_cards = self.state['hole_cards']
        board_cards = self.state['board_cards']
        rake = self.state['rake']
        for player in range(n_players):
            item = {
                'name': None,
                'role': None,
                'won': None,
                'hole': repr(hole_cards[player]).replace(",", ''),
                'rank': None,
                'fold': None
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
                item['won'] = int((1 - rake) * payouts[player])
            rank_val = self.env.judge.evaluate(hole_cards[player], board_cards)
            rank = self.env.judge.get_rank_class(rank_val)
            item['rank'] = rank
            item['fold'] = state['folded'][player]
            items.append(item)
        return items

    def fmt(self, history):
        out = ''
        betted = False
        for step, item in enumerate(history):
            state, player, action, info = item
            folded = state['folded']
            street = state['street']
            if folded[player] >= state['street']:
                out += f"agent_{player+1}: "
                if info["action_type"] == "fold":
                    out += "folds"
                elif info["action_type"] == "check":
                    out += "checks"
                elif info["action_type"] == "call":
                    out += f"calls ${action}"
                elif info["action_type"] in ["raise", "all_in"]:
                    if betted:
                        out += f"raises ${action - info['call']} to ${action}"
                    else:
                        if street:
                            out += f"bets ${action}"
                        else:
                            out += f"raises ${action} to ${info['call'] + action}"
                        betted = True
                else:
                    out += f"bets {action}"
                out += '\n'
        return out

# out += f"\t\t\t\t\taction {action} "
# out += f"call {info['call']} "
# out += f"min_raise {info['min_raise']} "
# out += f"max_raise {info['max_raise']} "
# out += f"stack {info['stack']} "
# out += f"pot {state['pot']} "
