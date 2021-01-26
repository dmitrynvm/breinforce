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

import pprint
pp = pprint.PrettyPrinter(indent=4)
np.random.seed(1)

def learn():
    env = gym.make('CustomSixPlayer-v0')
    probs = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
    players = [agents.RandomAgent(probs)] * 6
    env.register(players)
    obs = env.reset()

    while True:
        action = env.act(obs)
        obs, rewards, done = env.step(action)

        if all(done):
            break
    print(env.render())


if __name__ == "__main__":
    learn()
