import math
import torch
import numpy as np
from addict import Dict
from sanic import Sanic
from sanic import response
from breinforce import agents, models, views
import pprint; pp = pprint.PrettyPrinter(indent=4).pprint


class GreedyStrategy():
    def __init__(self, start, stop, decay):
        self.start = start
        self.stop = stop
        self.decay = decay

    def rate(self, step):
        return self.stop + (self.start - self.stop) * math.exp(-1. * step * self.decay)


strategy = GreedyStrategy(1, 0.01, 0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_nn = models.DQNetwork(387).to(device)
policy_nn.load_state_dict(torch.load('breinforce/api/policy_nn.pt'))
agent = agents.DQNAgent(strategy, 6, device)


app = Sanic('breinforce.api')


def smartpad(lst):
    if len(lst) < 6:
        out = lst + [0 for _ in range(6 - len(lst))]
    else:
        out = lst[:6]
    return out


def convert(body):
    body = Dict(body)
    streets = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    fold, call, min_raise, max_raise, allin = -1, 0, 0, 0, 0
    for action in body.valid_actions:
        if action.action == 'fold':
            fold = -1
        if action.action == 'call':
            call = action.amount
        if action.action == 'raise':
            min_raise = action.amount.min
            max_raise = action.amount.max
        if action.action == 'allin':
            allin = action.amount
    stacks = []
    for i, seat in enumerate(body.round_state.seats):
        if seat.state == 'participating':
            stacks += [seat.stack]
        if seat.name == 'ME':
            player = i
    stacks = smartpad(stacks)
    street = body.round_state.street
    history = body.round_state.action_histories[street]
    commits = []
    for episode in history:
        commits += [episode.amount]
    commits = smartpad(commits)
    folded = {}
    for street in body.round_state.action_histories:
        for episode in body.round_state.action_histories[street]:
            folded[episode.name] = 1
    folded = smartpad(list(folded.values()))
    alive = np.logical_not(folded).astype(int).tolist()

    obs = {
        'street': streets[body.round_state.street],
        'button': body.round_state.dealer_btn,
        'pot': body.round_state.pot.main.amount,
        'call': call,
        'min_raise': min_raise,
        'max_raise': max_raise,
        'stacks': stacks,
        'player': player,
        'community_cards': body.round_state.community_card,
        'hole_cards': body.hole_card,
        'alive': alive,
        'commits': commits,
        'valid_actions': {
            'fold': fold,
            'call': call,
            'raise': {
                'min': min_raise,
                'max': max_raise
            },
            'allin': allin
        }
    }
    return obs


@app.get('/')
async def index_get(request):
    return response.json({'server': 'works'})


@app.post('/')
async def index_post(request):
    obs = convert(request.json)
    action = agent.predict(obs, policy_nn)
    return response.json({'action': 'action', 'amount': action})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
