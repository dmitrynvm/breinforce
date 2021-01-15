"""Classes and functions to evaluate poker hands"""

import functools
import itertools
import operator
from typing import Dict, List
from breinforce import exceptions
from . import utils
from .card import Card


class HashMap:
    """Hash maps that transforms unique prime product of hands to unique
    integer hand rank. The lower the rank the better the hand

    Parameters
    ----------
    suits : int
        number of suits in deck
    ranks : int
        number of ranks in deck
    cards_for_hand : int
        number of cards used for a poker hand
    low_end_straight : bool, optional
        toggle to include straights where ace is the lowest card, by
        default True
    order : List[str], optional
        custom hand rank order, if None hands are ranked by rarity, by
        default None
    """

    ORDER_STRINGS = ["sf", "fk", "fh", "fl", "st", "tk", "tp", "pa", "hc"]

    def __init__(
        self,
        suits: int,
        ranks: int,
        cards_for_hand: int,
        low_end_straight: bool = True,
        order: List[str] = None,
    ):

        if order is not None:
            if any(string not in order for string in self.ORDER_STRINGS):
                raise exceptions.InvalidOrderError(
                    (
                        f"invalid order list {order},"
                        f"order list must be permutation \
                        of {self.ORDER_STRINGS}"
                    )
                )

        # number of suited and unsuited possibilities of different hands
        straight_flushes, u_straight_flushes = self.__straight_flush(
            suits, ranks, cards_for_hand, low_end_straight
        )
        four_of_a_kinds, u_four_of_a_kinds = self.__four_of_a_kind(
            suits, ranks, cards_for_hand
        )
        full_houses, u_full_houses = \
            self.__full_house(suits, ranks, cards_for_hand)
        flushes, u_flushes = self.__flush(
            suits, ranks, cards_for_hand, low_end_straight
        )
        straights, u_straights = self.__straight(
            suits, ranks, cards_for_hand, low_end_straight
        )
        three_of_a_kinds, u_three_of_a_kinds = self.__three_of_a_kind(
            suits, ranks, cards_for_hand
        )
        two_pairs, u_two_pairs = self.__two_pair(suits, ranks, cards_for_hand)
        pairs, u_pairs = self.__pair(suits, ranks, cards_for_hand)
        high_cards, u_high_cards = self.__high_card(
            suits, ranks, cards_for_hand, low_end_straight
        )

        self.hand_dict = {
            "straight flush": {
                "suited": straight_flushes,
                "unsuited": u_straight_flushes,
            },
            "four of a kind": {
                "suited": four_of_a_kinds,
                "unsuited": u_four_of_a_kinds,
            },
            "full house": {"suited": full_houses, "unsuited": u_full_houses},
            "flush": {"suited": flushes, "unsuited": u_flushes},
            "straight": {"suited": straights, "unsuited": u_straights},
            "three of a kind": {
                "suited": three_of_a_kinds,
                "unsuited": u_three_of_a_kinds,
            },
            "two pair": {"suited": two_pairs, "unsuited": u_two_pairs},
            "pair": {"suited": pairs, "unsuited": u_pairs},
            "high card": {"suited": high_cards, "unsuited": u_high_cards},
        }

        # suited hands
        s_hands = [
            (straight_flushes, "straight flush"),
            (four_of_a_kinds, "four of a kind"),
            (full_houses, "full house"),
            (flushes, "flush"),
            (straights, "straight"),
            (three_of_a_kinds, "three of a kind"),
            (two_pairs, "two pair"),
            (pairs, "pair"),
            (high_cards, "high card"),
        ]

        # sort suited hands and rank unsuited hands by suited
        # rank order or by order provided
        if order is None:
            s_hands = sorted(s_hands)
        else:
            idcs = [self.ORDER_STRINGS.index(hand) for hand in order]
            s_hands = [s_hands[idx] for idx in idcs]
        # lookup is done on unsuited hands but hand
        # rank is dependent on suited hands
        u_hands = [
            (self.hand_dict[uh[1]]["unsuited"], uh[1]) for uh in s_hands
        ]

        # compute cumulative number of unsuited hands for each hand
        # cumulative unsuited is the maximum rank a hand can have
        ranked_hands = []
        rank = 0
        cumulative_hands = 0
        for u_hand in u_hands:
            hand_rank = u_hand[1]
            cumulative_hands += u_hand[0]
            self.hand_dict[hand_rank]["cumulative unsuited"] = cumulative_hands
            if cumulative_hands > 0:
                self.hand_dict[hand_rank]["rank"] = rank
                rank += 1
                ranked_hands.append(hand_rank)
        self.max_rank = cumulative_hands

        # list of hands ordered by rank from best to worst
        self.ranked_hands = ranked_hands

        # create lookup tables
        self.suited_lookup: Dict[int, int] = {}
        self.unsuited_lookup: Dict[int, int] = {}
        self.__flushes(ranks, cards_for_hand, low_end_straight)
        self.__multiples(ranks, cards_for_hand)

        # if suited hands aren"t relevant set the suited
        # lookup table equal to the unsuited table
        if not self.hand_dict["flush"]["cumulative unsuited"]:
            self.suited_lookup = self.unsuited_lookup

    def lookup(self, cards: List[Card]) -> int:
        """Return unique hand rank for list of cards

        Parameters
        ----------
        cards : List[Card]
            list of cards to be evaluated

        Returns
        -------
        int
            hand rank
        """
        # if all flush bits equal then use flush lookup
        if functools.reduce(operator.and_, cards, 0xF000):
            hand_or = functools.reduce(operator.or_, cards) >> 16
            prime = utils.prime_product_from_rankbits(hand_or)
            return self.suited_lookup[prime]
        prime = utils.prime_product_from_hand(cards)
        return self.unsuited_lookup[prime]

    def __straight_flush(self, suits, ranks, cards_for_hand, low_end_straight):
        if cards_for_hand < 3 or suits < 2:
            return 0, 0
        # number of smallest cards which start straight
        unsuited = ranks - (cards_for_hand - 1) + low_end_straight
        # multiplied with number of suits
        suited = max(unsuited, unsuited * suits)
        return int(suited), int(unsuited)

    def __four_of_a_kind(self, suits, ranks, cards_for_hand):
        if cards_for_hand < 4 or suits < 4:
            return 0, 0
        # choose 1 rank for quads multiplied by
        # rank choice for remaining cards
        unsuited = \
            utils.ncr(ranks, 1) * utils.ncr(ranks - 1, cards_for_hand - 4)
        # mutliplied with number of suit choices for remaining cards
        suited = max(unsuited, unsuited * suits ** (cards_for_hand - 4))
        return int(suited), int(unsuited)

    def __full_house(self, suits, ranks, cards_for_hand):
        if cards_for_hand < 5 or suits < 3:
            return 0, 0
        # choose one rank for trips and pair multiplied by
        # rank choice for remaining cards
        unsuited = (
            utils.ncr(ranks, 1)
            * utils.ncr(ranks - 1, 1)
            * utils.ncr(ranks - 2, cards_for_hand - 5)
        )
        # multiplied with number of suit choices for
        # trips + pair and remaining cards
        suited = max(
            unsuited,
            unsuited * utils.ncr(suits, 3) * utils.ncr(suits, 2) * suits
            ** (cards_for_hand - 5),
        )
        return int(suited), int(unsuited)

    def __flush(self, suits, ranks, cards_for_hand, low_end_straight):
        if cards_for_hand < 3 or suits < 2:
            return 0, 0
        # all straight combinations
        straight_flushes = ranks - (cards_for_hand - 1) + low_end_straight
        # choose all cards from ranks minus straight flushes
        unsuited = utils.ncr(ranks, cards_for_hand) - straight_flushes
        # multiplied by number of suits
        suited = max(unsuited, unsuited * suits)
        return int(suited), int(unsuited)

    def __straight(self, suits, ranks, cards_for_hand, low_end_straight):
        if cards_for_hand < 3:
            return 0, 0
        # number of smallest cards which start straight
        unsuited = ranks - (cards_for_hand - 1) + low_end_straight
        # straight flush combinations
        straight_flushes = 0
        if suits > 1:
            straight_flushes = unsuited * suits
        # multiplied with suit choice for every card
        # minus straight flushes
        suited = max(
            unsuited,
            unsuited * suits ** cards_for_hand - straight_flushes
        )
        if suits < 2:
            suited = unsuited
        return int(suited), int(unsuited)

    def __three_of_a_kind(self, suits, ranks, cards_for_hand):
        if cards_for_hand < 3 or suits < 3:
            return 0, 0
        # choose one rank for trips multiplied by
        # rank choice for remaining cards
        unsuited = \
            utils.ncr(ranks, 1) * utils.ncr(ranks - 1, cards_for_hand - 3)
        # multiplied with suit choices for trips and remaining cards
        suited = max(
            unsuited,
            unsuited * utils.ncr(suits, 3) * utils.ncr(suits, 3)
            ** (cards_for_hand - 3)
        )
        return int(suited), int(unsuited)

    def __two_pair(self, suits, ranks, cards_for_hand):
        if cards_for_hand < 4 or suits < 2:
            return 0, 0
        # choose two ranks for pairs multiplied by
        # ranks for remaining cards
        unsuited = \
            utils.ncr(ranks, 2) * utils.ncr(ranks - 2, cards_for_hand - 4)
        # multiplied with suit choices for both pairs
        # and suit choices for remaining cards
        suited = max(
            unsuited,
            unsuited * utils.ncr(suits, 2) ** 2 * suits
            ** (cards_for_hand - 4)
        )
        return int(suited), int(unsuited)

    def __pair(self, suits, ranks, cards_for_hand):
        if cards_for_hand < 2 or suits < 2:
            return 0, 0
        # choose rank for pair multiplied by
        # ranks for remaining cards
        unsuited = \
            utils.ncr(ranks, 1) * utils.ncr(ranks - 1, cards_for_hand - 2)
        # multiplied with suit choices for pair and remaining cards
        suited = max(
            unsuited,
            unsuited * utils.ncr(suits, 2) * suits
            ** (cards_for_hand - 2)
        )
        return int(suited), int(unsuited)

    def __high_card(self, suits, ranks, cards_for_hand, low_end_straight):
        # number of smallest cards which start straight
        straights = 0
        if cards_for_hand > 2:
            straights = ranks - (cards_for_hand - 1) + low_end_straight
        # any combination of rank and subtract straights
        unsuited = utils.ncr(ranks, cards_for_hand) - straights
        # multiplied with suit choices for all cards
        # all same suits not allowed
        suited = max(unsuited, unsuited * (suits ** cards_for_hand - suits))
        if suits < 2:
            suited = unsuited
        return int(suited), int(unsuited)

    @staticmethod
    def __gen_straight_flush(cards_for_hand, ranks, low_end_straight):
        straight_flushes = []

        # start with best straight (flush)
        # for 5 card hand with 13 ranks: 0b1111100000000
        bin_n_str = "0b" + "1" * cards_for_hand + "0" * (13 - cards_for_hand)
        # remove one 0 for every straight (flush)
        for _ in range(ranks - (cards_for_hand - 1)):
            straight_flushes.append(int(bin_n_str, 2))
            bin_n_str = bin_n_str[:-1]
        if low_end_straight:
            # add low end straight
            bin_n_str = (
                "0b1"
                + "0" * (ranks - cards_for_hand)
                + "1" * (cards_for_hand - 1)
                + "0" * (13 - ranks)
            )
            straight_flushes.append(int(bin_n_str, 2))

        return straight_flushes

    @staticmethod
    def __gen_flush(cards_for_hand, ranks, straight_flushes):
        flushes = []
        # start with lowest non pair hand
        # for 5 card hand with 13 ranks: 0b11111
        bin_n_str = "0b" + ("1" * cards_for_hand)
        gen = utils.lexographic_next_bit(int(bin_n_str, 2))
        # iterate over all possibilities of unique hands
        for _ in range(int(utils.ncr(ranks, cards_for_hand))):
            # pull the next flush pattern from generator
            # offset by number of ranks not in play
            flush = next(gen) << (13 - ranks)
            if flush not in straight_flushes:
                flushes.append(flush)

        flushes.reverse()
        return flushes

    def __flushes(self, ranks, cards_for_hand, low_end_straight):
        straight_flushes = []
        if (
            self.hand_dict["straight flush"]["cumulative unsuited"]
            or self.hand_dict["straight"]["cumulative unsuited"]
        ):
            straight_flushes = self.__gen_straight_flush(
                cards_for_hand, ranks, low_end_straight
            )

        # dynamically generate all the other
        # flushes (including straight flushes)
        flushes = []
        if (
            self.hand_dict["flush"]["cumulative unsuited"]
            or self.hand_dict["high card"]["cumulative unsuited"]
        ):
            flushes = self.__gen_flush(cards_for_hand, ranks, straight_flushes)

        def add_to_map(rank_string, rank_bits, suited):
            if not self.hand_dict[rank_string]["cumulative unsuited"]:
                return
            n_ranks = len(rank_bits)
            assert n_ranks == self.hand_dict[rank_string]["unsuited"]
            hand_rank = self.__get_rank(rank_string)
            for rank_bit in rank_bits:
                prime_product = utils.prime_product_from_rankbits(rank_bit)
                if suited:
                    self.suited_lookup[prime_product] = hand_rank
                else:
                    self.unsuited_lookup[prime_product] = hand_rank
                hand_rank += 1

        add_to_map("straight flush", straight_flushes, True)
        add_to_map("flush", flushes, True)
        add_to_map("straight", straight_flushes, False)
        add_to_map("high card", flushes, False)

    def __multiples(self, ranks, cards_for_hand):
        def add_to_map(rank_string, multiples):
            # inverse ranks, A - 2
            backwards_ranks = list(range(13 - 1, 13 - 1 - ranks, -1))
            if not self.hand_dict[rank_string]["cumulative unsuited"]:
                return
            # get cumulative hand rank
            hand_rank = self.__get_rank(rank_string)
            # if different multiples (e.g. full house) order of
            # multiples is important
            if len(set(multiples)) > 1:
                multiple_combinations = itertools.permutations(
                    backwards_ranks, len(multiples)
                )
            # if same multiples (e.g. two pair) order of multiples
            # is unimportant
            else:
                multiple_combinations = itertools.combinations(
                    backwards_ranks, len(multiples)
                )
            for card_ranks in multiple_combinations:
                base_product = 1
                # compute product over every combination of multiple
                # card rank
                for card_rank, multiple in zip(card_ranks, multiples):
                    base_product *= Card.PRIMES[card_rank] ** multiple
                n_kickers = cards_for_hand - sum(multiples)
                # record product in lookup table
                if n_kickers:
                    kickers = backwards_ranks[:]
                    for card_rank in card_ranks:
                        kickers.remove(card_rank)
                    kicker_combinations = list(
                        itertools.combinations(kickers, n_kickers)
                    )
                    for kickers in kicker_combinations:
                        product = base_product
                        for kicker in kickers:
                            product *= Card.PRIMES[kicker]
                        self.unsuited_lookup[product] = hand_rank
                        hand_rank += 1
                else:
                    self.unsuited_lookup[base_product] = hand_rank
                    hand_rank += 1
            # check hand rank is equal to number of iterated ranks
            n_ranks = hand_rank - self.__get_rank(rank_string)
            assert n_ranks == self.hand_dict[rank_string]["unsuited"]

        add_to_map("four of a kind", [4])
        add_to_map("full house", [3, 2])
        add_to_map("three of a kind", [3])
        add_to_map("two pair", [2, 2])
        add_to_map("pair", [2])

    def __get_rank(self, hand):
        rank = self.hand_dict[hand]["rank"]
        if not rank:
            return 0
        better_hand = self.ranked_hands[rank - 1]
        return self.hand_dict[better_hand]["cumulative unsuited"] + 1
