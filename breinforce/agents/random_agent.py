import random
from typing import List
from . import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, probs: List[float]):
        super().__init__()
        self.probs = probs

    def act(self, obs):
        legal_actions = obs['legal_actions']
        return random.choice(legal_actions)
