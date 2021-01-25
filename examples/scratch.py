import gym
import math
import random
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from breinforce import agents, envs, utils
from tabulate import tabulate
from tqdm import tqdm
from time import sleep

np.random.seed(1)
pd.options.plotting.backend = "plotly"

Episode = namedtuple('Episode', ('state', 'action', 'next_state', 'reward'))

def learn():
    env = gym.make('CustomSixPlayer-v0')
    probs = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
    players = [agents.RandomAgent(probs)] * 6
    env.register(players)
    obs = env.reset()
    print(obs)
'''
    while True:
        action = env.act(obs)
        obs, rewards, done, info = env.step(action)

        if all(done):
            break

    print(env.history)
'''

if __name__ == "__main__":
    learn()
