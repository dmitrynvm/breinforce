from addict import Dict
from . import BaseAgent
from fractions import Fraction


class SplitAgent(BaseAgent):

    def __init__(self, splits):
        self.splits = splits

    def legal_actions(self, obs):
        valid_actions = obs['valid_actions']
        out = {}
        out['fold'] = valid_actions['fold']
        out['call'] = valid_actions['call']
        if 'raise' in valid_actions:
            raises = {}
            raise_min = obs['valid_actions']['raise']['min']
            raise_max = obs['valid_actions']['raise']['max']
            for split in self.splits:
                name = str(Fraction(split).limit_denominator())
                split = int(split * obs['pot'])
                if raise_min < split < raise_max:
                    raises[name] = split
            if raises:
                out['raise'] = raises
        out['allin'] = valid_actions['allin']
        return Dict(out)
