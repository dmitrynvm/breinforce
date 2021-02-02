import random
from breinforce.core.types import Action
from breinforce.core.utils import flatten
from . import SplitAgent


class RandomAgent(SplitAgent):

    def __init__(self, splits):
        super().__init__(splits)

    def predict(self, obs):
        legal_actions = flatten(self.legal_actions(obs))
        print(obs['valid_actions'])
        print(self.legal_actions(obs))
        action = random.choice(list(legal_actions.items()))
        return Action(*action)
