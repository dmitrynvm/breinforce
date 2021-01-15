import functools
import operator
from typing import List
from .card import Card


def lexographic_next_bit(bits):
    # generator next legographic bit sequence given a bit sequence with
    # N bits set e.g.
    # 00010011 -> 00010101 -> 00010110 -> 00011001 ->
    # 00011010 -> 00011100 -> 00100011 -> 00100101
    lex = bits
    yield lex
    while True:
        temp = (lex | (lex - 1)) + 1
        lex = temp | ((((temp & -temp) // (lex & -lex)) >> 1) - 1)
        yield lex


def ncr(n, r):
    r = min(r, n - r)
    numer = functools.reduce(operator.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(operator.mul, range(1, r + 1), 1)
    return numer / denom


def prime_product_from_rankbits(rankbits: int) -> int:
    """Computes prime product from rankbits of cards, primarily used
    for evaluating flushes and straights. Expects 13 bit integer
    with bits of the cards in the hand flipped.

    Parameters
    ----------
    rankbits : int
        13 bit integer with flipped rank bits

    Returns
    -------
    int
        prime product of rank cards
    """

    product = 1
    for i in Card.INT_RANKS:
        # if the ith bit is set
        if rankbits & (1 << i):
            product *= Card.PRIMES[i]
    return product


def prime_product_from_hand(cards: List[Card]) -> int:
    """Computes unique prime product for a list of cards. Used for
    evaluating hands

    Parameters
    ----------
    cards : List[Card]
        list of cards

    Returns
    -------
    int
        prime product of cards
    """

    product = 1
    for card in cards:
        product *= card & 0xFF
    return product
