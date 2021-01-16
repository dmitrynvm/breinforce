import random
from typing import List
from . import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, probs: List[float]):
        super().__init__()
        self.probs = probs

    def act(self, obs):
        legal_actions = obs['legal_actions']
        #print(legal_actions, self.probs)
        indices = list(range(len(legal_actions)))
        #index = random.choices(indices, self.probs)[0]
        index = random.choice(indices)
        #print('from', legal_actions, 'chosen', legal_actions[index])
        return legal_actions[index]
