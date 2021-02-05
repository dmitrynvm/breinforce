def select(history, street):
    out = []
    for episode in history:
        if episode.state['street'] == street:
            print(episode)
            print()
            out += [episode]
    return out


def header(episodes):
    out = ''
    state = episodes[0].state
    n_players = state.n_players
    hole_cards = state.hole_cards
    sb = state.small_blind
    bb = state.big_blind
    st = state.straddle
    sb_index = state.button + 1
    bb_index = state.button + 2
    st_index = state.button + 3
    sb_name = state.player_names[sb_index]
    bb_name = state.player_names[bb_index]
    st_name = state.player_names[st_index]

    date = state.date.strftime("%d %m %Y %H:%M:%S")
    out += f"***** 888.es Hand History for Game {state.game} *****\n"
    out += f"{state.small_blind}$ / {state.big_blind}$ "
    out += f"Blinds No Limit Holdem - *** {date}\n"
    out += f"Table {state.table} {state.n_players} Max (Real Money)\n"
    out += f"Seat {state.button + 1} is the button\n"
    out += f"Total number of players : {state.n_players}\n"

    for player, stack in enumerate(state.stacks):
        player_name = state.player_names[player]
        out += f'Seat {player+1}: {player_name} ({stack} $)\n'

    out += f'{sb_name}: posts small blind [{sb} $]\n'
    out += f'{bb_name}: posts small blind [{bb} $]\n'
    out += f'{st_name}: posts straddle [{st} $]\n'
    out += '** HOLE CARDS **\n'
    for player in range(n_players):
        player_name = state.player_names[player]
        player_cards = hole_cards[player]
        out += f'Dealt to {player_name} {player_cards}\n'
    return out


def verb(action_type, i):
    out = ''
    if 'fold' in action_type:
        out += 'folds'
    elif 'call' in action_type:
        out += 'calls'
    elif 'check' in action_type:
        out += 'checks'
    elif 'raise' in action_type or 'allin' in action_type:
        if i:
            out += 'raises'
        else:
            out += 'bets'
    else:
        out += "foo"
    return out


def preflop(history):
    history = select(history, 0)
    out = '** PREFLOP **\n'
    for i, episode in enumerate(history):
        state = episode.state
        action_name, action_value = episode.action
        action_type = action_name.split('_')[0]
        player_name = state.player_names[state.player]
        out += f"{player_name} {verb(action_type, i)} "
        state = episode.state
        if 'call' in action_type or 'raise' in action_type or 'allin' in action_type:
            out += f"[{action_value} $]"
        out += "\n"
    return out


def flop(history):
    history = select(history, 1)
    print(history)
    community_cards = history[-1].state.community_cards
    out = f'** FLOP ** {community_cards}\n'
    for i, episode in enumerate(history):
        state = episode.state
        action_name, action_value = episode.action
        action_type = action_name.split('_')[0]
        player_name = state.player_names[state.player]
        out += f"{player_name} {verb(action_type, i)} "
        if 'call' in action_type or 'raise' in action_type or 'allin' in action_type:
            out += f"[{action_value} $]"
        out += "\n"
    return out


def turn(history):
    history = select(history, 2)
    community_cards = history[0].state.community_cards[-1:]
    out = f'** TURN ** {community_cards}\n'
    for i, episode in enumerate(history):
        state = episode.state
        action_name, action_value = episode.action
        action_type = action_name.split('_')[0]
        player_name = state.player_names[state.player]
        out += f"{player_name} {verb(action_type, i)} "
        if 'call' in action_type or 'raise' in action_type or 'allin' in action_type:
            out += f"[{action_value} $]"
        out += "\n"
    return out


def river(history):
    history = select(history, 3)
    community_cards = history[0].state.community_cards[-1:]
    out = f'** RIVER ** {community_cards}\n'
    for i, episode in enumerate(history):
        state = episode.state
        action_name, action_value = episode.action
        print(episode.action)
        action_type = action_name.split('_')[0]
        player_name = state.player_names[state.player]
        out += f"{player_name} {verb(action_type, i)} "
        if 'call' in action_type or 'raise' in action_type or 'allin' in action_type:
            out += f"[{action_value} $]"
        out += "\n"
    return out


def summary(history):
    state = history[-1].state
    hole_cards = state.hole_cards
    payouts = state.payouts
    out = '** SUMMARY **\n'
    return out



def render(history):
    out = ''
    street = history[-1].state.street
    print('street', street)
    out += header(history)
    out += preflop(history)
    if street > 0:
        out += flop(history)
    if street > 1:
        out += turn(history)
    if street > 2:
        out += river(history)
    out += summary(history)
    return out
