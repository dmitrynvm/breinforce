import random
import pytest
from breinforce import exceptions
from breinforce.games.bropoker import Card, Deck


def test_draw():
    deck = Deck(2, 3)
    cards = deck.draw(1)
    assert len(cards) == 1
    cards = deck.draw(3)
    assert len(cards) == 3
    cards = deck.draw(4)
    assert len(cards) == 2
    cards = deck.draw(1)
    assert len(cards) == 0


def test_trick():
    random.seed(42)
    deck = Deck(4, 13)
    assert deck.cards[0] != deck.trick().cards[0]
    deck = Deck(4, 13).trick(['Ah', '2s'])
    cards = deck.draw(2)
    assert cards[0] == Card('Ah')
    assert cards[1] == Card('2s')
    deck = deck.shuffle()
    cards = deck.draw(2)
    assert cards[0] == Card('Ah')
    assert cards[1] == Card('2s')
    deck = deck.untrick().shuffle()
    cards = deck.draw(2)
    assert cards[0] != Card('Ah')
    assert cards[1] != Card('2s')


def test_invalid_init():
    with pytest.raises(exceptions.InvalidRankError):
        Card('1s')
    with pytest.raises(exceptions.InvalidRankError):
        Card('1t')
    with pytest.raises(exceptions.InvalidRankError):
        Deck(0, 0)
    with pytest.raises(exceptions.InvalidRankError):
        Deck(2, 14)
    with pytest.raises(exceptions.InvalidSuitError):
        Card('At')
    with pytest.raises(exceptions.InvalidSuitError):
        Deck(0, 1)
    with pytest.raises(exceptions.InvalidSuitError):
        Deck(5, 1)


def test_ops():
    card = Card('Ac')
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


def test_str_repr():
    # card = Card('Ac')
    # assert repr(card) == f'Card ({id(card)}): {card}'
    deck = Deck(4, 13)
    assert repr(deck) == f'Deck ({id(deck)}): {str(deck)}'
