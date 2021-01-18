"""Bropoker environment class for running poker games"""
from datetime import datetime
import gym
import numpy as np
from typing import Dict, List, Optional, Tuple
from breinforce.agents import BaseAgent
from breinforce.games.bropoker import Deck, Judge
from breinforce.views import AsciiView


def clean(legal_actions, action) -> int:
    """
    Find closest bet size to actual bet
    """
    index = np.argmin(np.absolute(np.array(legal_actions) - action))
    return legal_actions[index]


def get_call(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = state.commits.max() - state.commits[state.player]
        out = min(out, state.stacks[state.player])
    return out


def get_min_raise(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = max(state.straddle, get_call(state))
        out = min(out, state.stacks[state.player])
    return out


def get_max_raise(state) -> int:
    out = 0
    if not all(get_done(state)):
        out = min(state.stacks[state.player], state.raise_sizes[state.street])
    return out


def get_action_type(state, action):
    call = get_call(state)
    max_raise = get_max_raise(state)

    out = None
    if action == call and call == 0:
        out = "check"
    elif action == call and call > 0:
        out = "call"
    elif call < action < max_raise:
        out = "raise"
    elif action == max_raise:
        out = "all_in"
    else:
        out = "fold"
    return out


def get_done(state) -> List[bool]:
    if state.street >= state.n_streets or sum(state.alive) <= 1:
        return np.full(state.n_players, 1)
    return np.logical_not(state.alive)


def get_all_agreed(state) -> bool:
    if not all(state.acted):
        return False
    return all(
        (state.commits == state.commits.max())
        | (state.stacks == 0)
        | np.logical_not(state.alive)
    )


def get_legal_actions(state):
    call = get_call(state)
    max_raise = get_max_raise(state)
    outs = {0, call, max_raise}
    for split in state.splits:
        out = int(split * state.pot)
        if out < get_max_raise(state):
            outs.add(out)
    return list(sorted(outs))


def get_legal_actions_dict(state):
    max_raise = get_max_raise(state)
    out = {'fold': 0, 'call': get_call(state), 'all_in': max_raise}
    for i, split in enumerate(state.splits):
        action = int(split * state.pot)
        if action < max_raise:
            out[f"raise_{i+1}"] = action
    return out


def get_payouts(state) -> np.ndarray:
    # players that have folded lose their actions
    payouts = -1 * state.contribs * np.logical_not(state.alive)
    if sum(state.alive) == 1:
        payouts += state.alive * (state.pot - state.contribs)
    # if last street played and still multiple players alive
    elif state.street >= state.n_streets:
        print('HEEEE')
        payouts = evaluate(state)
        payouts -= state.contribs
    if any(payouts > 0):
        state.stacks += payouts + state.contribs
    return payouts


def get_observation(state) -> Dict:
    board_cards = [str(c) for c in state.board_cards]
    hole_cards = [[str(c) for c in cs] for cs in state.hole_cards]
    obs: dict = {
        "street": state.street,
        "button": state.button,
        "player": state.player,
        "pot": state.pot,
        "call": get_call(state),
        "max_raise": get_max_raise(state),
        "min_raise": get_min_raise(state),
        "legal_actions": get_legal_actions_dict(state),
        "board_cards": board_cards,
        "hole_cards": hole_cards,
        "alive": state.alive,
        "stacks": state.stacks,
        "commits": state.commits
    }
    return obs

def evaluate(state) -> np.ndarray:
    # grab array of hand strength and pot contribs
    worst_hand = state.judge.hashmap.max_rank + 1
    hand_list = []
    payouts = np.zeros(state.n_players, dtype=int)
    for player in range(state.n_players):
        # if not alive hand strength set
        # to 1 worse than worst possible rank
        hand_strength = worst_hand
        if state.alive[player]:
            hand_strength = state.judge.evaluate(
                state.hole_cards[player], state.board_cards
            )
        hand_list.append([player, hand_strength, state.contribs[player]])
    hands = np.array(hand_list)
    # sort hands by hand strength and pot contribs
    hands = hands[np.lexsort([hands[:, 2], hands[:, 1]])]
    pot = state.pot
    remainder = 0
    # iterate over hand strength and
    # pot contribs from smallest to largest
    for idx, (_, strength, contribs) in enumerate(hands):
        eligible = hands[:, 0][hands[:, 1] == strength].astype(int)
        # cut can only be as large as lowest player commit amount
        cut = np.clip(hands[:, 2], None, contribs)
        split_pot = sum(cut)
        split = split_pot // len(eligible)
        remain = split_pot % len(eligible)
        payouts[eligible] += split
        remainder += remain
        # remove chips from players and pot
        hands[:, 2] -= cut
        pot -= split_pot
        # remove player from next split pot
        hands[idx, 1] = worst_hand
        if pot == 0:
            break
    # give worst position player remainder chips
    if remainder:
        # worst player is first player after button involved in pot
        involved_players = np.nonzero(payouts)[0]
        button_shift = (involved_players <= state.button) * state.n_players
        button_shifted_players = involved_players + button_shift
        worst_idx = np.argmin(button_shifted_players)
        worst_pos = involved_players[worst_idx]
        payouts[worst_pos] += remainder
    return payouts


def move_(state):
    for idx in range(1, state.n_players + 1):
        player = (state.player + idx) % state.n_players
        if state.alive[player]:
            break
        else:
            state.acted[player] = True
    state.player = player


def perform_action_(state, action):
    action = min(state.stacks[state.player], action)
    state.pot += action
    state.contribs[state.player] += action
    state.commits[state.player] += action
    state.stacks[state.player] -= action
    state.acted[state.player] = True


def perform_antes_(state):
    actions = state.antes
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.contribs += actions
    state.stacks -= actions


def perform_blinds_(state):
    actions = state.blinds
    actions = np.roll(actions, state.player)
    actions = (state.stacks > 0) * state.alive * actions
    state.pot += sum(actions)
    state.commits += actions
    state.contribs += actions
    state.stacks -= actions



class Bropoker(gym.Env):

    def __init__(
            self,
            n_players: int,
            n_streets: int,
            n_suits: int,
            n_ranks: int,
            n_hole_cards: int,
            n_cards_for_hand: int,
            rake: float,
            raise_sizes: List[float],
            n_board_cards: List[int],
            blinds: List[int],
            antes: List[int],
            stacks: List[int],
            splits: List[str]
    ) -> None:

        # envs
        self.n_players = n_players
        self.n_streets = n_streets
        self.n_suits = n_suits
        self.n_ranks = n_ranks
        self.n_hole_cards = n_hole_cards
        self.n_cards_for_hand = n_cards_for_hand
        self.rake = rake
        self.raise_sizes = raise_sizes
        self.n_board_cards = n_board_cards
        self.blinds = np.array(blinds, dtype=int)
        self.antes = np.array(antes, dtype=int)
        self.splits = splits
        self.start_stacks = np.array(stacks, dtype=int)

        # agnt
        self.small_blind = blinds[0]
        self.big_blind = blinds[1] if n_players > 2 else None
        self.straddle = blinds[2] if n_players > 3 else None
        self.hand_id = "hand_1"
        self.date = datetime.now().strftime("%b/%d/%Y %H:%M:%S")
        self.table_name = "table_1"
        self.player_ids = ["agent_" + str(i+1) for i in range(self.n_players)]
        self.hole_cards = []
        self.board_cards = []
        self.payouts = None

        # dealer
        self.street = 0
        self.button = 0
        self.player = -1
        self.pot = 0
        self.alive = np.zeros(self.n_players, dtype=np.uint8)
        self.contribs = np.zeros(self.n_players, dtype=np.int32)
        self.stacks = np.array(stacks, dtype=np.int16)
        self.acted = np.zeros(self.n_players, dtype=np.uint8)
        self.commits = np.zeros(self.n_players, dtype=np.int32)
        self.history = []

        self.deck = Deck(self.n_suits, self.n_ranks)
        self.judge = Judge(n_suits, n_ranks, n_cards_for_hand)
        self.agents = None

    def register(self, agents: List) -> None:
        self.agents = agents

    def act(self, obs: dict) -> int:
        return self.agents[self.player].act(obs)

    def reset(self) -> Dict:
        """Resets the hash table. Shuffles the deck, deals new hole cards
        to all players, moves the button and collects blinds and antes.
        """
        self.alive.fill(1)
        self.stacks = self.start_stacks.copy()
        self.button = 0
        self.deck.shuffle()
        self.board_cards = self.deck.draw(self.n_board_cards[0])
        self.history = []
        self.hole_cards = [
            self.deck.draw(self.n_hole_cards) for _
            in range(self.n_players)
        ]
        self.pot = 0
        self.contribs.fill(0)
        self.street = 0
        self.commits.fill(0)
        self.acted.fill(0)
        self.player = self.button
        if self.n_players > 2:
            move_(self)
        perform_antes_(self)
        perform_blinds_(self)
        move_(self)
        move_(self)
        move_(self)
        return get_observation(self)

    def step(self, action: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        legal_actions = get_legal_actions(self)
        legal_actions_dict = get_legal_actions_dict(self)
        action = clean(legal_actions, action)
        action_type = get_action_type(self, action)
        call = get_call(self)
        info = {
            "action_type": action_type,
            "legal_actions": legal_actions,
            "call": call,
            "min_raise": get_min_raise(self),
            "max_raise": get_max_raise(self),
            "stack": self.stacks[self.player]
        }

        if call and ((action < call) or action < 0):
            self.alive[self.player] = 0
            action = 0

        perform_action_(self, action)
        self.history.append((self.state, self.player, action, info))
        move_(self)

        # if all agreed go to next street
        if get_all_agreed(self):
            self.player = self.button
            move_(self)
            # if at most 1 player alive and not all in turn up all
            # board cards and evaluate hand
            while True:
                self.street += 1
                allin = self.alive * (self.stacks == 0)
                all_allin = sum(self.alive) - sum(allin) <= 1
                if self.street >= self.n_streets:
                    break
                self.board_cards += self.deck.draw(
                    self.n_board_cards[self.street]
                )
                if not all_allin:
                    break
            self.commits.fill(0)
            self.acted = np.logical_not(self.alive).astype(np.uint8)

        obs = get_observation(self)
        payouts = get_payouts(self)
        done = get_done(self)
        info = None
        if all(done):
            self.player = -1
            obs["player"] = -1
        obs["hole_cards"] = obs["hole_cards"][obs["player"]]
        return obs, payouts, done, info

    def render(self, mode="ascii"):
        return "ascii"

    @property
    def state(self):
        out = {
            "table_name": self.table_name,
            "street": self.street,
            "sb": self.small_blind,
            "bb": self.big_blind,
            "st": self.straddle,
            "hand_id": self.hand_id,
            "date": self.date,
            "player_ids": self.player_ids,
            "hole_cards": self.hole_cards,
            "start_stacks": self.start_stacks,
            "player": self.player,
            "board_cards": self.board_cards,
            "button": self.button,
            "pot": self.pot,
            "payouts": get_payouts(self),
            "n_players": self.n_players,
            "n_hole_cards": self.n_hole_cards,
            "n_board_cards": sum(self.n_board_cards),
            "rake": self.rake,
            "antes": self.antes,
            "min_raise": get_min_raise(self),
            "max_raise": get_max_raise(self),
            "acted": self.acted.copy(),
            "alive": self.alive.copy(),
            "stacks": self.stacks.copy(),
            "commits": self.commits
        }
        return out
