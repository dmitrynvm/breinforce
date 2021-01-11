import random
from typing import List
from breinforce.agents import SplitAgent


class RandomSplitAgent(SplitAgent):

    def __init__(
        self,
        legal_actions: List[float],
        split: List[float],
        probs: List[float]
    ):
        super().__init__(legal_actions, split)
        self.probs = probs

    def act(self, obs):
        ids = list(range(len(self.legal_actions)))
        i = random.choices(ids, self.probs)[0]
        action = self.legal_actions[i]
        split = self.fractions[i]
        pot = obs['pot']
        call = obs['call']
        min_raise = obs['min_raise']
        max_raise = obs['max_raise']
        return super().act(action, split, pot, call, min_raise, max_raise)
