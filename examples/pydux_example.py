import pydux
from copy import deepcopy
from datetime import datetime
import json
import gym
import numpy as np
import os
import uuid
from typing import Dict, List, Optional, Tuple, Union
from breinforce import errors
from breinforce.agents import BaseAgent
from breinforce.config.application import CONFIG_DIR
from breinforce.games.bropoker import Card, Deck, Judge
from breinforce.views import AsciiView


def counter(state_, action):
    state = deepcopy(state_)
    if action['type'] == 'INCREMENT_COUNTER':
        state['public']['counter'] = state['public']['counter'] + 1
    elif action['type'] == 'DECREMENT_COUNTER':
        state['public']['counter'] = state['public']['counter'] - 1
    return state


initState = {
    "public": {
        "counter": 5,
    },
    "private": [
    ]
}

store = pydux.create_store(counter, initState)
store.subscribe(lambda: print(store.get_state()))
store.dispatch({'type': 'INCREMENT_COUNTER'})
store.dispatch({'type': 'INCREMENT_COUNTER'})
store.dispatch({'type': 'DECREMENT_COUNTER'})