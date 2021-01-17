import pytest
from breinforce import errors
from breinforce.games.bropoker import Card, Deck, HashMap, Judge


def test_init():
    with pytest.raises(errors.InvalidHandSizeError):
        Judge(4, 13, 0)
    with pytest.raises(errors.InvalidHandSizeError):
        Judge(4, 13, 6)
    with pytest.raises(errors.InvalidOrderError):
        HashMap(4, 13, 5, order=["lala"])


"""
def test_str_repr():
    judge = Judge(4, 13, 5)
    string = (
        "straight flush (0.0015%) > four of a kind (0.0240%) > "
        "full house (0.1441%) > flush (0.1965%) > straight (0.3925%) > "
        "three of a kind (2.1128%) > two pair (4.7539%) > "
        "pair (42.2569%) > high card (50.1177%)"
    )
    repr_string = (
        f"Judge ({id(judge)}): straight flush (0.0015%) > "
        f"four of a kind (0.0240%) > full house (0.1441%) > "
        f"flush (0.1965%) > straight (0.3925%) > "
        f"three of a kind (2.1128%) > two pair (4.7539%) > "
        f"pair (42.2569%) > high card (50.1177%)"
    )
    assert str(judge) == string
    assert repr(judge) == repr_string
"""


def test_hand_rank():
    judge = Judge(4, 13, 5)
    assert judge.get_rank_class(0) == "straight flush"
    assert judge.get_rank_class(7462) == "high card"
    with pytest.raises(errors.InvalidHandRankError):
        judge.get_rank_class(-1)
    with pytest.raises(errors.InvalidHandRankError):
        judge.get_rank_class(7463)


def test_1_card():
    # no board cards
    judge = Judge(1, 3, 1)
    hand1 = [Card("As")]
    hand2 = [Card("Ks")]
    assert judge.evaluate(hand1, []) < judge.evaluate(hand2, [])
    # 1 board card
    hand1 = [Card("As")]
    hand2 = [Card("Ks")]
    comm_cards = [Card("Qs")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    hand1 = [Card("Qs")]
    hand2 = [Card("Ks")]
    comm_cards = [Card("As")]
    assert judge.evaluate(hand1, comm_cards) == judge.evaluate(
        hand2, comm_cards
    )
    # 2 suits
    # 1 card for hand, no board cards
    judge = Judge(2, 3, 1)
    hand1 = [Card("Ah")]
    hand2 = [Card("As")]
    assert judge.evaluate(hand1, []) == judge.evaluate(hand2, [])


def test_2_card():
    # 1 suit
    judge = Judge(1, 3, 2)
    hand1 = [Card("Ks")]
    hand2 = [Card("Qs")]
    comm_cards = [Card("As")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # 2 suits
    # pair > high card
    judge = Judge(2, 3, 2)
    hand1 = [Card("Qs")]
    hand2 = [Card("Ks")]
    comm_cards = [Card("Qh")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # high card > low card
    hand1 = [Card("Ah")]
    hand2 = [Card("Ks")]
    comm_cards = [Card("Qs")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)


def test_3_card():
    # 1 suit
    # straight > high card
    judge = Judge(1, 13, 3)
    hand1 = [Card("Js")]
    hand2 = [Card("Qs")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # high card > low card
    hand1 = [Card("Ks")]
    hand2 = [Card("Qs")]
    comm_cards = [Card("5s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # ace high straight > ace low straight
    hand = [Card("As")]
    comm_cards1 = [Card("Qs"), Card("Ks")]
    comm_cards2 = [Card("2s"), Card("3s")]
    assert \
        judge.evaluate(hand, comm_cards1) < judge.evaluate(hand, comm_cards2)
    # 2 suits
    # straight flush > straight
    judge = Judge(2, 13, 3)
    hand1 = [Card("Js")]
    hand2 = [Card("Jc")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > pair
    hand1 = [Card("Jc")]
    hand2 = [Card("9c")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > flush
    hand1 = [Card("7c")]
    hand2 = [Card("As")]
    comm_cards = [Card("7s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > high card
    hand1 = [Card("9s")]
    hand2 = [Card("Ac")]
    comm_cards = [Card("7s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # 4 suits
    # straight flush > straight
    judge = Judge(4, 13, 3)
    hand1 = [Card("Js")]
    hand2 = [Card("Jc")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > pair
    hand1 = [Card("Jc")]
    hand2 = [Card("9c")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > pair
    hand1 = [Card("As")]
    hand2 = [Card("7c")]
    comm_cards = [Card("7s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("7c")]
    hand2 = [Card("Ac")]
    comm_cards = [Card("7s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)


def test_4_card():
    # 1 suit
    # straight > high card
    judge = Judge(1, 13, 4)
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("Qs"), Card("3s")]
    comm_cards = [Card("8s"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # high card > low card
    hand1 = [Card("Ks"), Card("2s")]
    hand2 = [Card("Qs"), Card("3s")]
    comm_cards = [Card("5s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # ace high straight > ace low straight
    hand = [Card("As")]
    comm_cards1 = [Card("Js"), Card("Qs"), Card("Ks")]
    comm_cards2 = [Card("2s"), Card("3s"), Card("4s")]
    assert \
        judge.evaluate(hand, comm_cards1) < judge.evaluate(hand, comm_cards2)
    # 2 suits
    # straight flush > two pair
    judge = Judge(2, 13, 4)
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("8h"), Card("9h")]
    comm_cards = [Card("8s"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # two pair > straight
    hand1 = [Card("8s"), Card("9h")]
    hand2 = [Card("Js"), Card("Ts")]
    comm_cards = [Card("8h"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > flush
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("4h"), Card("6h")]
    comm_cards = [Card("8h"), Card("9h")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > pair
    hand1 = [Card("8s"), Card("7s")]
    hand2 = [Card("Th"), Card("2s")]
    comm_cards = [Card("9s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("8s"), Card("7h")]
    hand2 = [Card("Th"), Card("2s")]
    comm_cards = [Card("8h"), Card("9h")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # 4 suits
    # four of a kind > straight flush
    judge = Judge(4, 13, 4)
    hand1 = [Card("As"), Card("Ac")]
    hand2 = [Card("Jh"), Card("Qh")]
    comm_cards = [Card("Kh"), Card("Ah"), Card("Ad")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight flush > three of a kind
    hand1 = [Card("Jh"), Card("Qh")]
    hand2 = [Card("As"), Card("Ac")]
    comm_cards = [Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # three of a kind > straight
    hand1 = [Card("As"), Card("Ac")]
    hand2 = [Card("Jd"), Card("Qd")]
    comm_cards = [Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > two pair
    hand1 = [Card("Jd"), Card("Qd")]
    hand2 = [Card("As"), Card("Kc")]
    comm_cards = [Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # two pair > flush
    hand1 = [Card("9h"), Card("Qh")]
    hand2 = [Card("8s"), Card("7s")]
    comm_cards = [Card("9s"), Card("Qs")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > pair
    hand1 = [Card("8s"), Card("7s")]
    hand2 = [Card("9h"), Card("2h")]
    comm_cards = [Card("9s"), Card("Qs")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("8s"), Card("7h")]
    hand2 = [Card("Ah"), Card("2s")]
    comm_cards = [Card("8h"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)


def test_5_card():
    # 1 suit
    # straight > high card
    judge = Judge(1, 13, 5)
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("Qs"), Card("3s")]
    comm_cards = [Card("7s"), Card("8s"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # high card > low card
    hand1 = [Card("Ks"), Card("2s")]
    hand2 = [Card("Qs"), Card("3s")]
    comm_cards = [Card("4s"), Card("5s"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # ace high straight > ace low straight
    hand = [Card("As")]
    comm_cards1 = [
        Card("Ts"),
        Card("Js"),
        Card("Qs"),
        Card("Ks"),
    ]
    comm_cards2 = [
        Card("2s"),
        Card("3s"),
        Card("4s"),
        Card("5s"),
    ]
    assert \
        judge.evaluate(hand, comm_cards1) < judge.evaluate(hand, comm_cards2)
    # 2 suits
    # straight flush > straight
    judge = Judge(2, 13, 5)
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("Jh"), Card("Th")]
    comm_cards = [Card("7s"), Card("8s"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > two pair
    hand1 = [Card("Js"), Card("Ts")]
    hand2 = [Card("7h"), Card("8h")]
    comm_cards = [Card("7s"), Card("8s"), Card("9s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # two pair > flush
    hand1 = [Card("7h"), Card("8h")]
    hand2 = [Card("Js"), Card("Ts")]
    comm_cards = [Card("7s"), Card("8s"), Card("2s")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > pair
    hand1 = [Card("8s"), Card("7s")]
    hand2 = [Card("Th"), Card("2s")]
    comm_cards = [Card("9s"), Card("Ts"), Card("As")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("8s"), Card("7h")]
    hand2 = [Card("Th"), Card("2s")]
    comm_cards = [Card("8h"), Card("9h"), Card("As")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # 3 suits
    # straight flush > four of a kind
    judge = Judge(3, 13, 5)
    hand1 = [Card("Jh"), Card("Qh")]
    hand2 = [Card("Kc"), Card("Kd")]
    comm_cards = [
        Card("Th"),
        Card("Kh"),
        Card("Ah"),
        Card("Ad"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # full house > three of a kind
    hand1 = [Card("Kc"), Card("Kd")]
    hand2 = [Card("Ac"), Card("Qh")]
    comm_cards = [
        Card("Th"),
        Card("Kh"),
        Card("Ah"),
        Card("Ad"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # three of a kind > straight
    hand1 = [Card("Ac"), Card("Qh")]
    hand2 = [Card("Qc"), Card("Jd")]
    comm_cards = [
        Card("Th"),
        Card("Kh"),
        Card("Ah"),
        Card("Ad"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > two pair
    hand1 = [Card("Jd"), Card("Td")]
    hand2 = [Card("Qd"), Card("Kd")]
    comm_cards = [Card("Qh"), Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # two pair > pair
    hand1 = [Card("Qd"), Card("Kd")]
    hand2 = [Card("Qc"), Card("Td")]
    comm_cards = [Card("Qh"), Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("8s"), Card("7h")]
    hand2 = [Card("Ah"), Card("2s")]
    comm_cards = [Card("8h"), Card("9h"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # 4 suits
    # straight flush > four of a kind
    judge = Judge(4, 13, 5)
    hand1 = [Card("Jh"), Card("Qh")]
    hand2 = [Card("As"), Card("Ac")]
    comm_cards = [
        Card("Th"),
        Card("Kh"),
        Card("Ah"),
        Card("Ad"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # four of a kind > full house
    hand1 = [Card("As"), Card("Ac")]
    hand2 = [Card("Kc"), Card("Kd")]
    comm_cards = [Card("Kh"), Card("Ah"), Card("Ad")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # full house > flush
    hand1 = [Card("Kc"), Card("Kd")]
    hand2 = [Card("Th"), Card("5h")]
    comm_cards = [
        Card("Kh"),
        Card("Ah"),
        Card("Ad"),
        Card("2h"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # flush > straight
    hand1 = [Card("Th"), Card("5h")]
    hand2 = [Card("Jd"), Card("Td")]
    comm_cards = [Card("Qh"), Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # straight > three of a kind
    hand1 = [Card("Jd"), Card("Td")]
    hand2 = [Card("Qd"), Card("Qc")]
    comm_cards = [Card("Qh"), Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # three of a kind > two pair
    hand1 = [Card("Qd"), Card("Qc")]
    hand2 = [Card("Kd"), Card("Ad")]
    comm_cards = [Card("Qh"), Card("Kh"), Card("Ah")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # two pair > pair
    hand1 = [Card("9h"), Card("Qh")]
    hand2 = [Card("8s"), Card("7s")]
    comm_cards = [Card("9s"), Card("Qs"), Card("8d")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
    # pair > high card
    hand1 = [Card("8s"), Card("7h")]
    hand2 = [Card("Ah"), Card("2s")]
    comm_cards = [Card("8h"), Card("9h"), Card("Ts")]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)


def test_short_deck():
    order = ["sf", "fk", "fl", "fh", "st", "tk", "tp", "pa", "hc"]
    judge = Judge(4, 9, 5, 0, order=order)
    # flush > full house
    hand1 = [Card("8h"), Card("7h")]
    hand2 = [Card("Jd"), Card("As")]
    comm_cards = [
        Card("Jh"),
        Card("9h"),
        Card("Ah"),
        Card("Ac"),
    ]
    assert \
        judge.evaluate(hand1, comm_cards) < judge.evaluate(hand2, comm_cards)
