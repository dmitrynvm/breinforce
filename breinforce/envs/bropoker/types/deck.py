import random
from .card import Card
from breinforce.core import errors


class Deck(object):
    """A deck contains at most 52 cards, 13 ranks 4 suits. Any "subdeck"
    of the standard 52 card deck is valid, i.e. the number of suits
    must be between 1 and 4 and number of ranks between 1 and 13. A
    deck can be tricked to ensure a certain order of cards.

    Parameters
    ----------
    n_suits : int
        number of suits to use in deck
    n_ranks : int
        number of ranks to use in deck
    """

    def __init__(self, n_suits: int, n_ranks: int) -> None:
        if n_ranks < 1 or n_ranks > 13:
            raise errors.InvalidRankError(
                f"Invalid number of suits, expected number of suits "
                f"between 1 and 13, got {n_ranks}"
            )
        if n_suits < 1 or n_suits > 4:
            raise errors.InvalidSuitError(
                f"Invalid number of suits, expected number of suits "
                f"between 1 and 4, got {n_suits}"
            )
        self.n_ranks = n_ranks
        self.n_suits = n_suits
        self.cards = []
        ranks = Card.STR_RANKS[-n_ranks:]
        suits = list(Card.SUITS_TO_INTS.keys())[:n_suits]
        for rank in ranks:
            for suit in suits:
                self.cards.append(Card(rank + suit))
        self.shuffle()

    def __str__(self) -> str:
        return "[{}]".format(",".join([str(card) for card in self.cards]))

    def __repr__(self) -> str:
        return "[{}]".format(",".join([repr(card) for card in self.cards]))

    def deal(self, n = 1):
        """Deal cards from the top of the deck. If the number of cards
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
        """
        out = []
        for _ in range(n):
            if self.cards:
                out.append(self.cards.pop(0))
            else:
                break
        return out

    def shuffle(self):
        """Shuffles the deck. If a tricking order is given, the desired
        cards are placed on the top of the deck after shuffling.

        Returns
        -------
        Deck
            self
        """
        random.shuffle(self.cards)

