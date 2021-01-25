import random
from typing import List
from . import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, probs: List[float]):
        super().__init__()
        self.probs = probs

    def act(self, obs):
        return random.choice(list(obs['legal_actions'].values()))
