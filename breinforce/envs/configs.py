
oo = float('inf')

BROPOKER = {
    'NolimitHoldemTwoPlayer-v0': {
        'n_players': 2,
        'n_streets': 4,
        'n_suits': 4,
        'n_ranks': 13,
        'n_hole_cards': 2,
        'n_cards_for_hand': 5,
        'rake': 0.0,
        'raise_sizes': [oo, oo, oo, oo],
        'ns_community_cards': [0, 3, 1, 1],
        'blinds': [1, 2],
        'antes': [0, 0],
        'stacks': [200, 200],
        'splits': [0.5, 1, 2]

    },
    'NolimitHoldemSixPlayer-v0': {
        'n_players': 6,
        'n_streets': 4,
        'n_suits': 4,
        'n_ranks': 13,
        'n_hole_cards': 2,
        'n_cards_for_hand': 5,
        'rake': 0.0,
        'raise_sizes': [oo, oo, oo, oo],
        'ns_community_cards': [0, 3, 1, 1],
        'blinds': [1, 2, 0, 0, 0, 0],
        'antes': [0, 0, 0, 0, 0, 0],
        'stacks': [200, 200, 200, 200, 200, 200],
        'splits': [0.5, 1, 2]

    },
    'NolimitHoldemNinePlayer-v0': {
        'n_players': 9,
        'n_streets': 4,
        'n_suits': 4,
        'n_ranks': 13,
        'n_hole_cards': 2,
        'n_cards_for_hand': 5,
        'rake': 0.0,
        'raise_sizes': [oo, oo, oo, oo],
        'ns_community_cards': [0, 3, 1, 1],
        'blinds': [1, 2, 0, 0, 0, 0, 0, 0, 0],
        'antes': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'stacks': [200, 200, 200, 200, 200, 200, 200, 200, 200],
        'splits': [0.5, 1, 2]

    },
    'CustomTwoPlayer-v0': {
        'n_players': 2,
        'n_streets': 4,
        'n_suits': 4,
        'n_ranks': 13,
        'n_hole_cards': 2,
        'n_cards_for_hand': 5,
        'rake': 0.0,
        'raise_sizes': [oo, oo, oo, oo],
        'ns_community_cards': [0, 3, 1, 1],
        'blinds': [1, 2],
        'antes': [0, 0],
        'stacks': [200, 200],
        'splits': [0.5, 1, 2]

    },
    'CustomSixPlayer-v0': {
        'n_players': 6,
        'n_streets': 4,
        'n_suits': 4,
        'n_ranks': 13,
        'n_hole_cards': 2,
        'n_cards_for_hand': 5,
        'rake': 0.05,
        'raise_sizes': [oo, oo, oo, oo],
        'ns_community_cards': [0, 3, 1, 1],
        'blinds': [1, 2, 4, 0, 0, 0],
        'antes': [1, 1, 1, 1, 1, 1],
        'stacks': [200, 200, 200, 200, 200, 200],
        'splits': [0.3, 0.5, 0.75, 1, 2]
    }
}