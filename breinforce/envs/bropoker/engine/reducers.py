from . import actions


def root(state, action):
    if action["type"] == "RESET":
        n_players = state['n_players']
        if n_players > 2:
            actions.move_player(state)
        #actions.collect_antes(state)
        actions.collect_blinds(state)
        actions.move_player(state)
        actions.move_player(state)
        actions.move_player(state)
    elif action["type"] == "STEP":
        actions.move_step(state, action['action'])
    return state
