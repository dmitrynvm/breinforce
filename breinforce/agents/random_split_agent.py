import random
from typing import List
from .split_agent import SplitAgent


class RandomSplitAgent(SplitAgent):

    def __init__(
        self,
        actions: List[float],
        fracs: List[float],
        probs: List[float]
    ):
        super().__init__(actions, fracs)
        self.probs = probs

    def act(self, obs):
        ids = list(range(len(self.actions)))
        i = random.choices(ids, self.probs)[0]
        action = self.actions[i]
        frac = self.fracs[i]
        pot = obs['pot']
        call = obs['call']
        min_raise = obs['min_raise']
        max_raise = obs['max_raise']
        return super().act(action, frac, pot, call, min_raise, max_raise)
