oo = float('inf')

KUHN_TWO_PLAYER = {
    'n_players': 2,
    'n_streets': 1,
    'n_suits': 1,
    'n_ranks': 3,
    'n_hole_cards': 1,
    'n_cards_for_hand': 1,
    'start_stack': 10,
    'n_community_cards': [0],
    'raise_sizes': [1],
    'n_raises': [1],
    'blinds': [0, 2],
    'antes': [1, 1],
}

KUHN_THREE_PLAYER = {
    'n_players': 3,
    'n_streets': 1,
    'n_suits': 1,
    'n_ranks': 3,
    'n_hole_cards': 1,
    'n_cards_for_hand': 1,
    'start_stack': 10,
    'n_community_cards': [0],
    'raise_sizes': [1],
    'n_raises': [1],
    'blinds': [0, 0, 0],
    'antes': [1, 1, 1],
}

LEDUC_TWO_PLAYER = {
    'n_players': 2,
    'n_streets': 2,
    'n_suits': 2,
    'n_ranks': 3,
    'n_hole_cards': 1,
    'n_cards_for_hand': 2,
    'start_stack': 10,
    'n_community_cards': [0, 1],
    'raise_sizes': [2, 2],
    'n_raises': [2, 2],
    'blinds': [0, 0],
    'antes': [1, 1],
}

LIMIT_HOLDEM_TWO_PLAYER = {
    'n_players': 2,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'n_community_cards': [0, 3, 1, 1],
    'raise_sizes': [2, 2, 4, 4],
    'n_raises': [3, 4, 4, 4],
    'blinds': [1, 2],
    'antes': [0, 0],
}

LIMIT_HOLDEM_SIX_PLAYER = {
    'n_players': 6,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'start_stack': 200,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'n_community_cards': [0, 3, 1, 1],
    'raise_sizes': [2, 2, 4, 4],
    'n_raises': [3, 4, 4, 4],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

LIMIT_HOLDEM_NINE_PLAYER = {
    'n_players': 9,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'start_stack': 200,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'raise_sizes': [2, 2, 4, 4],
    'n_raises': [3, 4, 4, 4],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

NOLIMIT_HOLDEM_TWO_PLAYER = {
    'n_players': 2,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2],
    'antes': [0, 0],
}

NOLIMIT_HOLDEM_SIX_PLAYER = {
    'n_players': 6,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

NOLIMIT_HOLDEM_NINE_PLAYER = {
    'n_players': 9,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

CUSTOM_TWO_PLAYER = {
    'n_players': 2,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2],
    'antes': [0, 0],
}

CUSTOM_SIX_PLAYER = {
    'n_players': 6,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

CUSTOM_NINE_PLAYER = {
    'n_players': 9,
    'n_streets': 4,
    'n_suits': 4,
    'n_ranks': 13,
    'n_hole_cards': 2,
    'n_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'n_raises': [oo, oo, oo, oo],
    'n_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}
