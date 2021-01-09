oo = float('inf')

KUHN_TWO_PLAYER = {
    'num_players': 2,
    'num_streets': 1,
    'num_suits': 1,
    'num_ranks': 3,
    'num_hole_cards': 1,
    'num_cards_for_hand': 1,
    'start_stack': 10,
    'num_community_cards': [0],
    'raise_sizes': [1],
    'num_raises': [1],
    'blinds': [0, 2],
    'antes': [1, 1],
}

KUHN_THREE_PLAYER = {
    'num_players': 3,
    'num_streets': 1,
    'num_suits': 1,
    'num_ranks': 3,
    'num_hole_cards': 1,
    'num_cards_for_hand': 1,
    'start_stack': 10,
    'num_community_cards': [0],
    'raise_sizes': [1],
    'num_raises': [1],
    'blinds': [0, 0, 0],
    'antes': [1, 1, 1],
}

LEDUC_TWO_PLAYER = {
    'num_players': 2,
    'num_streets': 2,
    'num_suits': 2,
    'num_ranks': 3,
    'num_hole_cards': 1,
    'num_cards_for_hand': 2,
    'start_stack': 10,
    'num_community_cards': [0, 1],
    'raise_sizes': [2, 2],
    'num_raises': [2, 2],
    'blinds': [0, 0],
    'antes': [1, 1],
}

LIMIT_HOLDEM_TWO_PLAYER = {
    'num_players': 2,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'num_community_cards': [0, 3, 1, 1],
    'raise_sizes': [2, 2, 4, 4],
    'num_raises': [3, 4, 4, 4],
    'blinds': [1, 2],
    'antes': [0, 0],
}

LIMIT_HOLDEM_SIX_PLAYER = {
    'num_players': 6,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'start_stack': 200,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'num_community_cards': [0, 3, 1, 1],
    'raise_sizes': [2, 2, 4, 4],
    'num_raises': [3, 4, 4, 4],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

LIMIT_HOLDEM_NINE_PLAYER = {
    'num_players': 9,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'start_stack': 200,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'raise_sizes': [2, 2, 4, 4],
    'num_raises': [3, 4, 4, 4],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

NOLIMIT_HOLDEM_TWO_PLAYER = {
    'num_players': 2,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2],
    'antes': [0, 0],
}

NOLIMIT_HOLDEM_SIX_PLAYER = {
    'num_players': 6,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

NOLIMIT_HOLDEM_NINE_PLAYER = {
    'num_players': 9,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

CUSTOM_TWO_PLAYER = {
    'num_players': 2,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2],
    'antes': [0, 0],
}

CUSTOM_SIX_PLAYER = {
    'num_players': 6,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0],
}

CUSTOM_NINE_PLAYER = {
    'num_players': 9,
    'num_streets': 4,
    'num_suits': 4,
    'num_ranks': 13,
    'num_hole_cards': 2,
    'num_cards_for_hand': 5,
    'start_stack': 200,
    'raise_sizes': [oo, oo, oo, oo],
    'num_raises': [oo, oo, oo, oo],
    'num_community_cards': [0, 3, 1, 1],
    'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
    'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}
