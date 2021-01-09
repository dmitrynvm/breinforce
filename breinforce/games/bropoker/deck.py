import random
from typing import List, Optional, Union
from .card import Card
from breinforce import errors


class Deck:
    '''A deck contains at most 52 cards, 13 ranks 4 suits. Any 'subdeck'
    of the standard 52 card deck is valid, i.e. the number of suits
    must be between 1 and 4 and number of ranks between 1 and 13. A
    deck can be tricked to ensure a certain order of cards.

    Parameters
    ----------
    num_suits : int
        number of suits to use in deck
    num_ranks : int
        number of ranks to use in deck
    '''

    def __init__(self, num_suits: int, num_ranks: int) -> None:
        if num_ranks < 1 or num_ranks > 13:
            raise errors.InvalidRankError(
                f'Invalid number of suits, expected number of suits '
                f'between 1 and 13, got {num_ranks}'
            )
        if num_suits < 1 or num_suits > 4:
            raise errors.InvalidSuitError(
                f'Invalid number of suits, expected number of suits '
                f'between 1 and 4, got {num_suits}'
            )
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.full_deck: List[Card] = []
        ranks = Card.STR_RANKS[-num_ranks:]
        suits = list(Card.SUITS_TO_INTS.keys())[:num_suits]
        for rank in ranks:
            for suit in suits:
                self.full_deck.append(Card(rank + suit))
        self._tricked = False
        self._top_idcs: List[int] = []
        self._bottom_idcs: List[int] = []
        self.shuffle()

    def __str__(self) -> str:
        string = ','.join([str(card) for card in self.cards])
        string = f'[{string}]'
        return string

    def __repr__(self) -> str:
        return f'Deck ({id(self)}): {str(self)}'

    def draw(self, n: int = 1) -> List[Card]:
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
        cards = []
        for _ in range(n):
            if self.cards:
                cards.append(self.cards.pop(0))
            else:
                break
        return cards

    def shuffle(self) -> 'Deck':
        '''Shuffles the deck. If a tricking order is given, the desired
        cards are placed on the top of the deck after shuffling.

        Returns
        -------
        Deck
            self
        '''
        self.cards = list(self.full_deck)
        if self._tricked and self._top_idcs and self._bottom_idcs:
            top_cards = [self.full_deck[idx] for idx in self._top_idcs]
            bottom_cards = [self.full_deck[idx] for idx in self._bottom_idcs]
            random.shuffle(bottom_cards)
            self.cards = top_cards + bottom_cards
        else:
            random.shuffle(self.cards)
        return self

    def trick(
        self,
        top_cards: Optional[List[Union[str, Card]]] = None,
        shuffle: bool = True
    ) -> 'Deck':
        '''Tricks the deck by placing a fixed order of cards on the top
        of the deck and shuffling the rest. E.g.
        deck.trick(['AS', '2H']) places the ace of spades and deuce of
        hearts on the top of the deck. The order of tricked cards
        persists even after untricking. That is, calling
        deck.trick(...).untrick().trick() will keep the deck tricked
        in the order given in the first trick call.

        Parameters
        ----------
        top_cards : Optional[List[Union[str, Card]]], optional
            list of cards to be placed on the top of the deck, by
            default None, by default None
        shuffle : bool, optional
            shuffles the deck after tricking, by default True

        Returns
        -------
        Deck
            self
        '''
        if top_cards is None and not self._top_idcs:
            self._tricked = False
            return self.shuffle()
        if top_cards:
            cards = [
                Card(top_card) if isinstance(top_card, str) else top_card
                for top_card in top_cards
            ]
            self._top_idcs = [self.full_deck.index(c) for c in cards]
            all_idcs = set(range(self.num_ranks * self.num_suits))
            self._bottom_idcs = list(all_idcs.difference(set(self._top_idcs)))
        self._tricked = True
        return self.shuffle()

    def untrick(self) -> 'Deck':
        '''Removes the tricked cards from the top of the deck. The order
        of tricked cards persists even after untricking. That is,
        calling deck.trick(...).untrick().trick() will keep the deck
        tricked in the order given in the first trick call.

        Returns
        -------
        Deck
            self
        '''
        self._tricked = False
        return self
