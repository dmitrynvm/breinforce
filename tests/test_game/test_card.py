import random
import pytest
from breinforce import errors
from breinforce.games.bropoker import Card


def test_invalid_init():
    with pytest.raises(errors.InvalidRankError):
        Card("1s")
    with pytest.raises(errors.InvalidRankError):
        Card("1t")

def test_ops():
    card = Card("Ac")
    assert card & card
    assert card & card.value
    assert card | card
    assert card | card.value
    assert card << 0 == card.value
    assert card.value << 0 == card.value
    assert card >> 0 == card.value
    assert card.value >> 0 == card.value
    with pytest.raises(NotImplementedError):
        assert card == 0

