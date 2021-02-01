import random
import pytest
from breinforce.core import errors
from breinforce.games.bropoker import Card


def test_ops():
    card = Card('Ac')
    assert card & card
    assert card & card.value
    assert card | card


def test_init():
    card1 = Card('Ac')
    card2 = Card('ac')
    card3 = Card('ca')
    card4 = Card('Ca')
    card5 = Card('cA')
    card6 = Card('CA')

    assert card1 == card2 == card3 == \
        card4 == card5 == card6
