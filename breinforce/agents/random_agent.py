import random
from typing import List
from . import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, probs):
        super().__init__()
        self.probs = probs

    def act(self, obs):
        print('agent', obs.legal_actions)
        return random.choice(list(obs.legal_actions.values()))
