import random
from . import Card


def create(n_suits, n_ranks):
    '''A deck contains at most 52 cards, 13 ranks 4 suits. Any 'subdeck'
    of the standard 52 card deck is valid, i.e. the number of suits
    must be between 1 and 4 and number of ranks between 1 and 13. A
    deck can be tricked to ensure a certain order of cards.

    Parameters
    ----------
    n_suits : int
        number of suits to use in deck
    n_ranks : int
        number of ranks to use in deck
    '''
    out = []
    ranks = Card.STR_RANKS[-n_ranks:]
    suits = list(Card.SUITS_TO_INTS.keys())[:n_suits]
    for rank in ranks:
        for suit in suits:
            out.append(Card(rank + suit))
    return out


def shuffle(cards):
    '''Shuffles the deck. If a tricking order is given, the desired
    cards are placed on the top of the deck after shuffling.

    Returns
    -------
    Deck
        self
    '''
    return random.sample(cards, len(cards))


def deal(cards, n = 1):
    '''Draws cards from the top of the deck. If the number of cards
    to draw exceeds the number of cards in the deck, all cards
    left in the deck are returned.

    Parameters
    ----------
    n : int, optional
        number of cards to draw, by default 1

    Returns
    -------
    List[Card]
        cards drawn from the deck
    '''
    out = []
    for _ in range(n):
        if cards:
            out.append(cards.pop(0))
        else:
            break
    return out
