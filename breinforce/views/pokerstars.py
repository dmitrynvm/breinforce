import os
from breinforce import games


def select(history, street):
    return [e for e in history if e.state['street'] == street]


def pretty(history):
    out = ''
    has_betted = False
    for step, episode in enumerate(history):
        state, player, action, info = episode
        folded = state['folded']
        street = state['street']
        out += f'agent_{player+1}: '
        if 'fold' == action.name:
            out += 'folds'
        elif not state['valid_actions']['call']:
            out += f'checks'
        elif 'call' == action.name:
            out += f'calls ${action.value}'
        elif state['valid_actions']['call'] == state['valid_actions']['allin']:
            out += f'calls ${action.value}'
        else:
            if has_betted:
                out += f"raises ${action.value - state['valid_actions']['call']} to ${action.value}"
            else:
                if street:
                    out += f"raises ${action.value - state['valid_actions']['call']} to ${action.value}"
                else:
                    out += f'bets ${action.value}'
                has_betted = True
        out += '\n'
    return out


def header(history):
    out = ''
    state = history[0].state
    game = state['game']
    date = state['date']
    small_blind = state['small_blind']
    big_blind = state['big_blind']
    straddle = state['straddle']
    button = state['button']
    table = state['table']
    n_players = state['n_players']
    player_names = state['player_names']
    start_stacks = state['start_stacks']
    # Game
    out += f'PokerStars Hand #{game}: Hold\'em No Limit' \
        f'(${small_blind}/${big_blind}/${straddle} chips) - {date} MSK\n'
    # Table
    out += f'Table \'{table}\' {n_players}-max ' \
        f'Seat #{button + 1} is the button\n'
    # Seats
    for i, start_stack in enumerate(start_stacks):
        player_name = player_names[i]
        out += f'Seat {i+1}: {player_name} (${start_stack} in chips)\n'
    return out


def preflop(history):
    # Preflop
    out = ''
    state = history[0].state
    button = state['button']
    small_blind = state['small_blind']
    big_blind = state['big_blind']
    straddle = state['straddle']
    n_players = state['n_players']
    player_names = state['player_names']
    small_blind_name = player_names[button + 1]
    big_blind_name = player_names[button + 2]
    hole_cards = state['hole_cards']
    out += f'{small_blind_name}: posts small blind ${small_blind}\n'
    out += f'{big_blind_name}: posts big blind ${big_blind}\n'
    if n_players > 3:
        straddle_name = player_names[button + 3]
        out += f'{straddle_name}: posts straddle ${straddle}\n'
    # Dealt
    out += '*** HOLE CARDS ***\n'
    for player in range(n_players):
        player_name = player_names[player]
        player_cards = repr(hole_cards[player]).replace(",", '')
        out += f'Dealt to {player_name} {player_cards}\n'

    episodes = select(history, 0)
    out += pretty(episodes)
    return out


def flop(history):
    out = ''
    state = history[0].state
    community_cards = state['community_cards']
    flop_cards = repr(community_cards[:3]).replace(",", '') if len(community_cards) > 2 else None
    episodes = select(history, 1)
    if episodes:
        out += f'*** FLOP *** {flop_cards}\n'
        out += pretty(episodes)
    return out


def turn(history):
    out = ''
    state = history[0].state
    community_cards = state['community_cards']
    turn_cards = repr(community_cards[:3]).replace(",", '') + '[' + repr(community_cards[3]) + ']' if len(community_cards) > 3 else None
    episodes = select(history, 2)
    if episodes:
        out += f'*** TURN *** {turn_cards}\n'
        out += pretty(episodes)
    return out


def river(history):
    out = ''
    state = history[0].state
    community_cards = state['community_cards']
    river_cards = repr(community_cards[:4]).replace(",", '') + '[' + repr(community_cards[4]).replace(",", '') + ']' if len(community_cards) > 4 else None
    river = select(history, 3)
    if river:
        out += f'*** RIVER *** {river_cards}\n'
        out += pretty(river)
    return out


def summary(history):
    out = ''
    state = history[-1].state
    n_players = state['n_players']
    pot = state['pot']
    rake = state['rake']
    community_cards = state['community_cards']

    results = get_results(state)
    out += '*** SUMMARY ***\n'
    out += f'Total pot ${pot} | rake ${int(pot * rake)} \n'
    cards = repr(community_cards).replace(",", '')
    out += f'Board {cards}\n'

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


def get_results(state):
    items = []
    n_players = state['n_players']
    n_players = state['n_players']
    payouts = state['payouts']
    hole_cards = state['hole_cards']
    community_cards = state['community_cards']
    rake = state['rake']
    for player in range(n_players):
        item = {
            'name': None,
            'role': None,
            'won': None,
            'hole': repr(hole_cards[player]).replace(",", ''),
            'rank': None,
            'fold': None
        }
        item['name'] = state['player_names'][player]
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

        judge = games.bropoker.Judge(4, 13, 5)
        rank_val = judge.evaluate(hole_cards[player], community_cards)
        rank = judge.get_rank_class(rank_val)
        item['rank'] = rank
        item['fold'] = state['folded'][player]
        items.append(item)
    return items


def render(history):
    '''Render representation based on the table configuration
    '''
    out = ''
    out += header(history)
    out += preflop(history)
    out += flop(history)
    out += turn(history)
    out += river(history)
    out += summary(history)
    return out
