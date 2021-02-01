import random
from . import SplitAgent
from breinforce.core.types import Action
from breinforce.core.utils import flatten
from breinforce.games.bropoker import Card, Judge


class RuleBasedAgent(SplitAgent):

    def __init__(self, splits):
        super().__init__(splits)
        self.judge = Judge(4, 13, 5)

    def predict(self, obs):
        legal_actions = self.legal_actions(obs)
        legal_raises = legal_actions['raise']
        if obs['street'] == 0:
            action = ('call', legal_actions['call'])
        elif obs['street'] in [1, 2]:
            if legal_raises:
                action = random.choice(list(legal_raises.items()))
            else:
                action = ('call', legal_actions['call'])
        elif obs['street'] == 3:
            community_cards = [Card(c) for c in obs['community_cards']]
            hole_cards = [Card(c) for c in obs['hole_cards']]
            if self.judge.evaluate(hole_cards, community_cards) > 2000:
                action = ('allin', legal_actions['allin'])
            else:
                if 'check' in legal_actions:
                    action = ('check', legal_actions['check'])
                else:
                    action = ('fold', legal_actions['fold'])
        return Action(*action)
