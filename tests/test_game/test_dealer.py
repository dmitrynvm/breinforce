import random
import pytest
from breinforce import envs, errors
from breinforce.games.bropoker import Dealer


def test_limit_bet_size():
    config = envs.configs.LIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 2.1
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 5
    assert sum(obs['street_commits']) == obs['pot']
    bet = 10
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 9
    assert sum(obs['street_commits']) == obs['pot']
    bet = 6
    _ = dealer.step(bet)
    bet = 8
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 23
    assert obs['max_raise'] == 0
    assert obs['call'] == 7
    bet = 7
    obs, *_ = dealer.step(bet)
    bet = -1
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 30
    assert not obs['active'].all()


def test_allin_bet_size():
    config = envs.configs.NOLIMIT_HOLDEM_TWO_PLAYER
    dealer = Dealer(**config)
    dealer.stacks[0] -= 150
    dealer.stacks[1] += 150
    obs = dealer.reset(reset_button=True, reset_stacks=False)
    bet = 100
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 52
    bet = 1000
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 400


def test_incomplete_raise():
    config = dict(**envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER)
    dealer = Dealer(**config)
    dealer.stacks[1] = dealer.stacks[1] - 190
    dealer.stacks[2] = dealer.stacks[2] + 190
    obs = dealer.reset(reset_button=True, reset_stacks=False)
    bet = -1
    _ = dealer.step(bet)
    _ = dealer.step(bet)
    _ = dealer.step(bet)
    bet = 8
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 11
    assert obs['call'] == 7
    assert obs['min_raise'] == 9
    assert obs['max_raise'] == 9
    bet = 9
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 20
    assert obs['call'] == 8
    assert obs['min_raise'] == 14  # call 8 + 6 largest valid raise
    bet = 8
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 28
    assert obs['call'] == 2
    assert obs['min_raise'] == 0
    assert obs['max_raise'] == 0


def test_bet_rounding():
    config = envs.configs.NOLIMIT_HOLDEM_NINE_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 1
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][3] == 0
    bet = 6
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][4] == 6
    bet = 3
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][5] == 0
    assert not obs['active'][5]
    bet = 4
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][6] == 6
    bet = 8
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][7] == 6
    bet = 9
    obs, *_ = dealer.step(bet)
    assert obs['street_commits'][8] == 10


def test_big_blind_raise_chance():
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 2  # all call
    for _ in range(5):
        obs, *_ = dealer.step(bet)
    assert obs['player'] == 2
    assert obs['call'] == 0
    assert obs['min_raise'] == 2


def test_all_but_one_fold():
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset(reset_button=True, reset_stacks=True)
    bet = -1
    for _ in range(5):
        obs, payouts, done = dealer.step(bet)
    assert all(done)
    assert obs['pot'] == 3
    test_payouts = [0, -1, 1, 0, 0, 0]
    assert all(
        payout == test_payout for payout,
        test_payout in zip(payouts, test_payouts)
    )
    test_stacks = [200, 199, 201, 200, 200, 200]
    assert all(
        stack == test_stack for stack, test_stack
        in zip(obs['stacks'], test_stacks)
    )


def test_all_allin():
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 200
    for _ in range(6):
        obs, payouts, done = dealer.step(bet)
    assert all(done)
    assert obs['pot'] == 1200
    test_payouts = [-200, -200, -200, -200, -200, 1000]
    assert all(
        payout == test_payout for payout,
        test_payout in zip(payouts, test_payouts)
    )
    test_stacks = [0, 0, 0, 0, 0, 1200]
    assert all(
        stack == test_stack for stack, test_stack
        in zip(obs['stacks'], test_stacks)
    )


def test_bet_after_round_end():
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 200
    for _ in range(6):
        obs, payouts, done = dealer.step(bet)
    assert all(done)
    assert obs['call'] == obs['min_raise'] == obs['max_raise'] == 0
    assert obs['player'] == -1
    obs, payouts, done = dealer.step(bet)
    assert all(done)
    assert obs['call'] == obs['min_raise'] == obs['max_raise'] == 0
    assert obs['player'] == -1


def test_too_few_players():
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 200
    for _ in range(6):
        obs, payouts, done = dealer.step(bet)
    assert all(done)
    assert obs['pot'] == 1200
    test_payouts = [-200, -200, -200, -200, -200, 1000]
    assert all(
        payout == test_payout for payout, test_payout
        in zip(payouts, test_payouts)
    )
    test_stacks = [0, 0, 0, 0, 0, 1200]
    assert all(
        stack == test_stack for stack, test_stack
        in zip(obs['stacks'], test_stacks)
    )
    with pytest.raises(errors.TooFewActivePlayersError):
        dealer.reset()


def test_button_move():
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_TWO_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset(reset_button=True, reset_stacks=False)
    assert obs['button'] == 0
    assert obs['player'] == 0
    bet = 0
    while True:
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    obs = dealer.reset(reset_button=False, reset_stacks=True)
    assert obs['button'] == 1
    assert obs['player'] == 1
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset(reset_button=True, reset_stacks=True)
    assert obs['button'] == 0
    assert obs['player'] == 3
    bet = 0
    while True:
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    obs = dealer.reset(reset_button=False, reset_stacks=True)
    assert obs['button'] == 1
    assert obs['player'] == 4


def test_inactive_players():
    random.seed(42)
    config = envs.configs.NOLIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 200
    _ = dealer.step(bet)
    bet = 200
    _ = dealer.step(bet)
    bet = -1
    while True:
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    obs = dealer.reset(reset_button=False, reset_stacks=False)
    assert obs['button'] == 1
    assert obs['player'] == 5


def test_game():
    config = envs.configs.LEDUC_TWO_PLAYER
    dealer = Dealer(**config)
    dealer.deck = dealer.deck.trick(['Qs', 'Ks', 'Qh'])
    _ = dealer.reset(reset_button=True, reset_stacks=True)
    bet = 2
    _ = dealer.step(bet)
    bet = 4
    _ = dealer.step(bet)
    bet = 2
    _ = dealer.step(bet)
    bet = 0
    _ = dealer.step(bet)
    bet = 2
    _ = dealer.step(bet)
    bet = 2
    obs, payout, done = dealer.step(bet)
    assert all(done)
    assert payout[0] > payout[1]
    assert payout[0] == 7


def test_heads_up():
    config = envs.configs.NOLIMIT_HOLDEM_TWO_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset(reset_button=True, reset_stacks=True)
    assert obs['player'] == 0
    assert obs['call'] == 1
    assert obs['min_raise'] == 3
    assert obs['max_raise'] == 199
    bet = 1
    obs, *_ = dealer.step(bet)
    assert obs['call'] == 0
    assert obs['min_raise'] == 2
    assert obs['max_raise'] == 198


def test_init_step():
    config = envs.configs.NOLIMIT_HOLDEM_TWO_PLAYER
    dealer = Dealer(**config)
    with pytest.raises(errors.HashMapResetError):
        dealer.step(0)


def test_split_pot():
    config = envs.configs.NOLIMIT_HOLDEM_NINE_PLAYER
    dealer = Dealer(**config)
    hands = [
        ['6c', '8s'],
        ['Ac', 'Ad'],
        ['Kd', '2h'],
        ['Th', '9c'],
        ['Js', 'Jc'],
        ['6h', '8d'],
        ['5c', '7d'],
        ['Qh', '2c'],
        ['3d', '4s'],
    ]
    comm_cards = ['4d', '5h', '7c', 'Ac', 'Kh']
    top_cards = [card for hand in hands for card in hand] + comm_cards
    dealer.deck = dealer.deck.trick(top_cards)
    obs = dealer.reset(reset_button=True, reset_stacks=True)
    bet = -1
    _ = dealer.step(bet)
    bet = 5
    _ = dealer.step(bet)
    bet = 5
    _ = dealer.step(bet)
    bet = 5
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = 5
    _ = dealer.step(bet)
    bet = 4
    _ = dealer.step(bet)
    bet = -1
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 27
    # flop
    bet = 4
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = 4
    _ = dealer.step(bet)
    bet = 4
    _ = dealer.step(bet)
    bet = 4
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 43
    while True:
        bet = 0
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    assert not sum(payouts)
    test_payouts = [12, -9, -2, 0, -5, 13, -9, 0, 0]
    assert all(
        payout == test_payout for payout,
        test_payout in zip(payouts, test_payouts)
    )


def test_allin():
    config = envs.configs.NOLIMIT_HOLDEM_NINE_PLAYER
    dealer = Dealer(**config)
    hands = [
        ['6c', '8s'],
        ['Ac', 'Ad'],
        ['Kd', '2h'],
        ['Th', '9c'],
        ['Js', 'Jc'],
        ['6h', '8d'],
        ['5c', '7d'],
        ['Qh', '2c'],
        ['3d', '4s'],
    ]
    comm_cards = ['4d', '5h', '7c', 'Ac', 'Kh']
    top_cards = [card for hand in hands for card in hand] + comm_cards
    dealer.deck = dealer.deck.trick(top_cards)
    dealer.stacks[0] = dealer.stacks[0] - 180
    dealer.stacks[1] = dealer.stacks[1] + 180
    obs = dealer.reset(reset_button=True, reset_stacks=False)
    bet = -1
    _ = dealer.step(bet)
    bet = 50
    _ = dealer.step(bet)
    bet = 0
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = 20
    _ = dealer.step(bet)
    bet = 49
    _ = dealer.step(bet)
    bet = -1
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 122
    while True:
        bet = 0
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    assert not sum(payouts)
    test_payouts = [42, 10, -2, 0, -50, 0, 0, 0, 0]
    assert all(
        payout == test_payout for payout,
        test_payout in zip(payouts, test_payouts)
    )


def test_allin_split_pot():
    config = envs.configs.NOLIMIT_HOLDEM_NINE_PLAYER
    dealer = Dealer(**config)
    hands = [
        ['6c', '8s'],
        ['Ac', 'Ad'],
        ['Kd', '2h'],
        ['Th', '9c'],
        ['6d', '8h'],
        ['6h', '8d'],
        ['5c', '7d'],
        ['Qh', '2c'],
        ['3d', '4s'],
    ]
    comm_cards = ['4d', '5h', '7c', 'Ac', 'Kh']
    top_cards = [card for hand in hands for card in hand] + comm_cards
    dealer.deck = dealer.deck.trick(top_cards)
    dealer.stacks[0] = dealer.stacks[0] - 180
    dealer.stacks[1] = dealer.stacks[1] + 180
    dealer.stacks[5] = dealer.stacks[5] - 165
    dealer.stacks[7] = dealer.stacks[7] + 165
    obs = dealer.reset(reset_button=True, reset_stacks=False)
    bet = -1
    _ = dealer.step(bet)
    bet = 45
    _ = dealer.step(bet)
    bet = 35
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = -1
    _ = dealer.step(bet)
    bet = 20
    _ = dealer.step(bet)
    bet = 44
    _ = dealer.step(bet)
    bet = -1
    obs, *_ = dealer.step(bet)
    assert obs['pot'] == 147
    while True:
        bet = 0
        obs, payouts, done = dealer.step(bet)
        if all(done):
            break
    # main pot 82 (27 per person, 1 remainder)
    # first side pot 45 (22 per person, 1 remainder)
    # second side pot 20
    # [27-20, -45, -2, 0, 27+22+20+2-45, 27+22-35, 0, 0, 0]
    assert not sum(payouts)
    test_payouts = [7, -45, -2, 0, 26, 14, 0, 0, 0]
    assert all(
        payout == test_payout for payout,
        test_payout in zip(payouts, test_payouts)
    )
