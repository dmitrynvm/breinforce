from breinforce.envs.bropoker.types import Card, Judge
import collections
import torch
from addict import Dict
from fractions import Fraction


def get_score(obs):

    community_cards = [Card(c) for c in obs['community_cards']]
    hole_cards = [Card(c) for c in obs['hole_cards']]
    judge = Judge(4, 13, 5)
    score = judge.evaluate(hole_cards, community_cards)

    return score


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_current(policy_nn, states, actions):
    return policy_nn(states).gather(dim=1, index=actions.unsqueeze(-1))


def get_next(target_nn, next_states):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_nn(non_final_states).max(dim=1)[0].detach()
    return values


def legal_actions(obs, splits):
    valid_actions = obs['valid_actions']
    out = {}
    out['fold'] = valid_actions['fold']
    out['call'] = valid_actions['call']
    if 'raise' in valid_actions:
        raises = {}
        raise_min = obs['valid_actions']['raise']['min']
        raise_max = obs['valid_actions']['raise']['max']
        for split in splits:
            name = str(Fraction(split).limit_denominator())
            split = int(split * obs['pot'])
            if raise_min < split < raise_max:
                raises[name] = split
        if raises:
            out['raise'] = raises
    out['allin'] = valid_actions['allin']
    return Dict(out)


def get_legal_action(obs, action, a_w_types):

    action_type = action.numpy()[0]

    if a_w_types[action_type] not in obs['legal_actions']:
        exist_raise = False
        for a in obs['legal_actions']:
            if 'raise' in a:
                exist_raise = True

        if exist_raise is True:
            for r in ['raise_1/3', 'raise_1/2', 'raise_3/4', 'raise_1', 'raise_3/2']:
                if r in obs['legal_actions']:
                    inv_map = {v: k for k, v in a_w_types.items()}
                    action = torch.tensor([inv_map[r]])
                    action_sum = obs['legal_actions'][r]
                    break
        else:
            if action_type == 1 or action_type == 2 or action_type == 3 or action_type == 4:
                action = torch.tensor([0])
                action_sum = obs['legal_actions']['fold']
            else:
                action = torch.tensor([7])
                action_sum = obs['legal_actions']['allin']
    else:
        if action_type == 0:
            action_sum = obs['legal_actions']['fold']
        elif action_type == 7:
            action_sum = obs['legal_actions']['allin']
        elif action_type == 6:
            action_sum = obs['legal_actions']['call']
        elif action_type == 1:
            action_sum = obs['legal_actions']['raise_1/3']
        elif action_type == 2:
            action_sum = obs['legal_actions']['raise_1/2']
        elif action_type == 3:
            action_sum = obs['legal_actions']['raise_3/4']
        elif action_type == 4:
            action_sum = obs['legal_actions']['raise_1']
        else:
            action_sum = obs['legal_actions']['raise_3/2']

    return action, action_sum