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
        if 'call' in valid_actions:
            out['call'] = valid_actions['call']
        if 'raise' in valid_actions:
            raises = {}
            raise_min = obs['valid_actions']['raise']['min']
            raise_max = obs['valid_actions']['raise']['max']
            for i, frac in enumerate(self.splits):
                name = 'raise_' + str(Fraction(frac).limit_denominator())
                split = int(frac * obs['pot'])
                if raise_min < split < raise_max:
                    raises[name] = split
            out['raise'] = raises
        out['allin'] = valid_actions['allin']
        return Dict(out)
