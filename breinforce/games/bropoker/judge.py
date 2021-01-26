"""Classes and functions to evaluate poker hands"""

import itertools
from typing import List
from breinforce import errors
from .card import Card
from .hashmap import HashMap







class Judge(object):
    """Evalutes poker hands using hole and board cards

    Parameters
    ----------
    suits : int
        number of suits in deck
    ranks : int
        number of ranks in deck
    cards_for_hand : int
        number of cards used for valid poker hand
    low_end_straight : bool, optional
        toggle to include the low ace straight within valid hands, by
        default True
    order : list, optional
        optional custom order of hand ranks, must be permutation of
        ["sf", "fk", "fh", "fl", "st", "tk", "tp", "pa", "hc"]. if
        order=None, hands are ranked by rarity. by default None
    """

    def __init__(
            self,
            suits: int,
            ranks: int,
            cards_for_hand: int,
            low_end_straight: bool = False,
            order: list = None,
    ):

        if cards_for_hand < 1 or cards_for_hand > 5:
            raise errors.InvalidHandSizeError(
                f"Evaluation for {cards_for_hand} "
                f"card hands is not supported. "
                f"bropoker currently supports 1-5 card poker hands"
            )

        self.suits = suits
        self.ranks = ranks
        self.cards_for_hand = cards_for_hand

        self.hashmap = HashMap(
            suits,
            ranks,
            cards_for_hand,
            low_end_straight=low_end_straight,
            order=order
        )

        hand_dict = self.hashmap.hand_dict
        total = sum(
            hand_dict[hand]["suited"]
            for hand in self.hashmap.ranked_hands
        )

        hands = [
            "{} ({:.4%})".format(hand, hand_dict[hand]["suited"] / total)
            for hand in self.hashmap.ranked_hands
        ]
        self.hand_ranks = " > ".join(hands)

    def __str__(self) -> str:
        return self.hand_ranks

    def __repr__(self) -> str:
        return f"Judge ({id(self)}): {str(self)}"

    def evaluate(self, hole_cards: List[Card], board_cards: List[Card]) -> int:
        """Evaluates the hand rank of a poker hand from a list of hole
        and a list of board cards. Empty hole and board cards
        are supported as well as requiring a minimum number of hole
        cards to be used.

        Parameters
        ----------
        hole_cards : List[Card]
            list of hole cards
        board_cards : List[Card]
            list of board cards

        Returns
        -------
        int
            hand rank
        """
        all_card_combs = list(
            itertools.combinations(
                hole_cards + board_cards, self.cards_for_hand
            )
        )

        minimum = self.hashmap.max_rank

        for card_comb in all_card_combs:
            senvs = self.hashmap.lookup(list(card_comb))
            if senvs < minimum:
                minimum = senvs
        return minimum

    def get_rank_class(self, hand_rank: int) -> str:
        """Outputs hand rank string from integer hand rank

        Parameters
        ----------
        hand_rank : int
            hand_rank (int): integer hand rank

        Returns
        -------
        str
            hand rank string
        """
        if hand_rank < 0 or hand_rank > self.hashmap.max_rank:
            raise errors.InvalidHandRankError(
                (
                    f"invalid hand rank, expected 0 <= hand_rank"
                    f" <= {self.hashmap.max_rank}, got {hand_rank}"
                )
            )
        hand_dict = self.hashmap.hand_dict
        for hand in self.hashmap.ranked_hands:
            if hand_rank <= hand_dict[hand]["cumulative unsuited"]:
                return hand
        raise errors.InvalidHandRankError(
            (
                f"invalid hand rank, expected 0 <= hand_rank"
                f" <= {self.hashmap.max_rank}, got {hand_rank}"
            )
        )
