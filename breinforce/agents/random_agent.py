import random
from typing import List
from . import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, probs: List[float]):
        super().__init__()
        self.probs = probs

    def act(self, obs):
        legal_actions = obs['legal_actions']
        indices = list(range(len(legal_actions)))
        index = random.choice(indices)
        out = legal_actions[index]
        '''
        if obs['street'] == 0:
            out = legal_actions['call']
        elif obs['street'] in [1, 2]:
            action_type = random.choice(['raise_1', 'raise_2'])
            if action_type in legal_actions:
                out = legal_actions[action_type]
            else:
                out = legal_actions['call']
        elif obs['street'] == 3:
            action_type = random.choice(['fold', 'all_in'])
            if action_type in legal_actions:
                out = legal_actions[action_type]
            else:
                out = 0
        # print(legal_actions, out)
        '''
        return out
