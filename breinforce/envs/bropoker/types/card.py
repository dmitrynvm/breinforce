'''Classes and functions to create and manipulate cards and lists of
cards from a standard 52 card poker deck'''

from breinforce.core import errors


class Card(object):
    '''Cards are represented as 32-bit integers. Most of the bits are used
    and have a specific meaning, check the poker README for details:

                  bitrank  suit rank   prime
        +--------+--------+--------+--------+
        |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
        +--------+--------+--------+--------+

        1) p = prime number of rank (deuce=2,trey=3,four=5,...,ace=41)
        2) r = rank of card (deuce=0,trey=1,four=2,five=3,...,ace=12)
        3) cdhs = suit of card (bit turned on based on suit of card)
        4) b = bit turned on depending on rank of card
        5) x = unused

    Parameters
    ----------
    string : str
        card string of format '{rank}{suit}' where rank is from
        [2-9, T/t, J/j, Q/q, K/k, A/a] and suit is from
        [S/s, H/h, D/d, C/c]

    Examples
    ------
    Card('TC'), Card('7H'), Card('ad')...
    '''

    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['S', 'H', 'D', 'C']
    STR_RANKS = '23456789TJQKA'
    INT_RANKS = list(range(13))
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    RANKS_TO_INTS = dict(zip(list(STR_RANKS), INT_RANKS))
    SUITS_TO_INTS = {
        'S': 1,  # spades
        'H': 2,  # hearts
        'D': 4,  # diamonds
        'C': 8,  # clubs
    }

    RAW_SUITS = {
        1: 's',  # spades
        2: 'h',  # hearts
        4: 'd',  # diamonds
        8: 'c',  # clubs
    }

    PRT_SUITS = {
        1: chr(9824),  # spades
        2: chr(9829),  # hearts
        4: chr(9830),  # diamonds
        8: chr(9827),  # clubs
    }

    def __init__(self, string: str) -> None:

        rank_char, suit_char = string.upper()
        rank_cond = rank_char in Card.RANKS
        suit_cond = suit_char in Card.SUITS
        if not rank_cond or not suit_cond:
            rank_char, suit_char = suit_char, rank_char

        rank_int = Card.RANKS_TO_INTS[rank_char]
        suit_int = Card.SUITS_TO_INTS[suit_char]
        rank_prime = Card.PRIMES[rank_int]

        bitrank = 1 << rank_int << 16
        suit = suit_int << 12
        rank = rank_int << 8

        self.value = bitrank | suit | rank | rank_prime

    def __str__(self) -> str:
        suit_int = (self.value >> 12) & 0xF
        rank_int = (self.value >> 8) & 0xF
        suit = Card.RAW_SUITS[suit_int]
        rank = Card.STR_RANKS[rank_int]
        return f'{rank}{suit}'

    def __repr__(self) -> str:
        suit_int = (self.value >> 12) & 0xF
        rank_int = (self.value >> 8) & 0xF
        suit = Card.RAW_SUITS[suit_int]
        rank = Card.STR_RANKS[rank_int]
        return f'{rank}{suit}'

    def __and__(self, other):
        if isinstance(other, Card):
            other = other.value
        return self.value & other

    def __rand__(self, other):
        return other & self.value

    def __or__(self, other):
        if isinstance(other, Card):
            other = other.value
        return self.value | other

    def __ror__(self, other):
        return other | self.value

    def __lshift__(self):
        return self.value << other

    def __rshift__(self, other):
        return self.value >> other

    def __eq__(self, other):
        if isinstance(other, Card):
            return bool(self.value == other.value)
        raise NotImplementedError('only comparisons of two cards allowed')
