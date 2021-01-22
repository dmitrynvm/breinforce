import random
from typing import List
from . import BaseAgent
from breinforce.games.bropoker import Card, Judge


class RulebAgent(BaseAgent):

    def __init__(self, probs: List[float]):
        super().__init__()
        self.probs = probs
        self.judge = Judge(4, 13, 5)

    def act(self, obs):
        legal_actions = obs['legal_actions']
        hole_cards = [Card(c) for c in obs['hole_cards']]
        board_cards = [Card(c) for c in obs['board_cards']]
        if obs['street'] == 0:
            out = legal_actions['call']
        elif obs['street'] in [1, 2]:
            action_type = random.choice(['raise_1', 'raise_2'])
            if action_type in legal_actions:
                out = legal_actions[action_type]
            else:
                out = legal_actions['call']
        elif obs['street'] == 3:
            if self.judge.evaluate(hole_cards, board_cards) > 2000:
                out = legal_actions["all_in"]
            else:
                out = legal_actions["fold"]
        return out
