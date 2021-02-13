import numpy as np
from breinforce.envs.bropoker.types import Judge


def observe(state):
    community_cards = [str(c) for c in state.community_cards]
    hole_cards = [[str(c) for c in cs] for cs in state.hole_cards]
    out = {
        'street': state.street,
        'button': state.button,
        'player': state.player,
        'pot': state.pot,
        'community_cards': community_cards,
        'hole_cards': hole_cards[state.player],
        'alive': state.alive.tolist(),
        'stacks': state.stacks.tolist(),
        'commits': state.commits.tolist(),
        'valid_actions': get_valid_actions(state)
    }
    return out


def reward(state):
    # players that have folded lose their actions
    out = -1 * state.contribs * np.logical_not(state.alive)
    if sum(state.alive) == 1:
        out += state.alive * (state.pot - state.contribs)
    # if last street played and still multiple players alive
    elif state.street >= state.n_streets:
        out = evaluate(state)
        out -= state.contribs
    if any(out > 0):
        state.stacks += out + state.contribs
    return out


def evaluate(state):
    judge = Judge(state.n_suits, state.n_ranks, state.n_cards_for_hand)
    # grab array of hand strength and pot contribs
    worst_hand = judge.hashmap.max_rank + 1
    hand_list = []
    rewards = np.zeros(state.n_players, dtype=int)
    for player in range(state.n_players):
        # if not alive hand strength set
        # to 1 worse than worst possible rank
        hand_strength = worst_hand
        if state.alive[player]:
            hand_strength = judge.evaluate(
                state.hole_cards[player], state.community_cards
            )
        hand_list.append([player, hand_strength, state.contribs[player]])
    hands = np.array(hand_list)
    # sort hands by hand strength and pot contribs
    hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
    pot = state.pot
    remainder = 0
    # iterate over hand strength and
    # pot contribs from smallest to largest
    for idx, (_, strength, contribs) in enumerate(hands):
        eligible = hands[:, 0][hands[:, 1] == strength].astype(int)
        # cut can only be as large as lowest player commit amount
        cut = np.clip(hands[:, 2], None, contribs)
        split_pot = sum(cut)
        split = split_pot // len(eligible)
        remain = split_pot % len(eligible)
        rewards[eligible] += split
        remainder += remain
        # remove chips from players and pot
        hands[:, 2] -= cut
        pot -= split_pot
        # remove player from move split pot
        hands[idx, 1] = worst_hand
        if pot == 0:
            break
    # give worst position player remainder chips
    if remainder:
        # worst player is first player after button involved in pot
        involved_players = np.nonzero(rewards)[0]
        button_shift = (involved_players <= state.button) * state.n_players
        button_shifted_players = involved_players + button_shift
        worst_idx = np.argmin(button_shifted_players)
        worst_pos = involved_players[worst_idx]
        rewards[worst_pos] += remainder
    return rewards


def agree(state):
    max_commit = state.commits.max()
    acted = state.acted == 1
    empty = state.stacks == 0
    allin = state.commits == max_commit
    folded = np.logical_not(state.alive)
    return acted * (empty + allin + folded)


def done(state):
    if state.street >= state.n_streets or sum(state.alive) <= 1:
        return np.full(state.n_players, 1)
    return np.logical_not(state.alive)


def results(state):
    return observe(state), reward(state), done(state)


def get_valid_actions(state):
    out = {}
    call = state.commits.max() - state.commits[state.player]
    call = min(call, state.stacks[state.player])
    raise_min = max(state.straddle, call + state.largest)
    raise_min = min(raise_min, state.stacks[state.player])
    street = state.street % state.n_streets
    raise_max = min(state.stacks[state.player], state.raise_sizes[street])
    out['fold'] = 0
    out['call'] = call
    out['raise'] = {
        'min': raise_min,
        'max': raise_max
    }
    out['allin'] = raise_max
    return out


def collect_action(state, action):
    action_value = max(0, min(state.stacks[state.player], action.value))
    state.pot += action_value
    state.contribs[state.player] += action_value
    state.commits[state.player] += action_value
    state.stacks[state.player] -= action_value
    state.acted[state.player] = 1


def collect_antes(state):
    actions = state.antes
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.contribs += actions
    state.stacks -= actions


def collect_blinds(state):
    actions = state.blinds
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.commits += actions
    state.contribs += actions
    state.stacks -= actions


def update_largest(state, action):
    valid_actions = get_valid_actions(state)
    if action.value and (action.value - valid_actions['call']) >= state.largest:
        state.largest = action.value - valid_actions['call']


def update_folded(state, action):
    valid_actions = get_valid_actions(state)
    if 'call' in valid_actions:
        if valid_actions['call'] and ((action.value < valid_actions['call']) or action.value < 0):
            state.alive[state.player] = 0
            state.folded[state.player] = state.street
    else:
        state.alive[state.player] = 0
        state.folded[state.player] = state.street


def update_rewards(state):
    state.rewards = reward(state)


def update_valid_actions(state):
    state.valid_actions = get_valid_actions(state)


def move_player(state):
    for idx in range(1, state.n_players + 1):
        player = (state.player + idx) % state.n_players
        if state.alive[player]:
            break
        else:
            state.acted[player] = True
    state.player = player


def move_street(state):
    if all(agree(state)):
        #state.player = state.button
        # if at most 1 player alive and not all in turn up all
        # board cards and evaluate hand
        while True:
            state.street += 1#(state.street + 1) % state.n_streets
            allin = state.alive * (state.stacks == 0)
            all_allin = sum(state.alive) - sum(allin) <= 1
            if state.street >= state.n_streets:
                break
            state.community_cards += state.deck.deal(
                state.ns_community_cards[state.street]
            )
            if not all_allin:
                break
        state.commits.fill(0)
        state.acted = np.logical_not(state.alive).astype(int)


def move_step(state, action):
    update_largest(state, action)
    update_folded(state, action)
    update_valid_actions(state)
    collect_action(state, action)
    move_player(state)
    move_street(state)
    update_rewards(state)


def reset(state):
    n_players = state['n_players']
    if n_players > 2:
        actions.move_player(state)
    #collect_antes(state)
    collect_blinds(state)
    move_player(state)
    move_player(state)
    move_player(state)
