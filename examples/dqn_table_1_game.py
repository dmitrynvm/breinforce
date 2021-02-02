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
from breinforce import agents, core, envs
from tabulate import tabulate
import plotly.graph_objects as go
from tqdm import tqdm
from time import sleep

np.random.seed(1)
pd.options.plotting.backend = 'plotly'

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class SequentialMemory():
    def __init__(self, size):
        self.size = size
        self.curr = 0
        self.items = []

    def add(self, item):
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            self.items[self.curr % self.size] = item
        self.curr += 1

    def extract(self, experiences):
        batch = Experience(*zip(*experiences))
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.stack(batch.reward, axis=0)
        t4 = torch.cat(batch.next_state)
        return (t1, t2, t3, t4)

    def sample(self, n_items):
        return self.extract(random.sample(self.items, n_items))

    def ready(self, n_items):
        return n_items < len(self.items)


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


class GreedyStrategy():
    def __init__(self, start, stop, decay):
        self.start = start
        self.stop = stop
        self.decay = decay

    def rate(self, step):
        return self.stop + (self.start - self.stop) * math.exp(-1. * step * self.decay)


class DQNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=100)
        self.fc5 = nn.Linear(in_features=100, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=6)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return x


class DQNAgent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def encode(self, obs):
        pads = ['--' for i in range(5 - len(obs['community_cards']))]
        street = obs['street']
        button = obs['button']
        player = obs['player']
        pot = obs['pot']
        call = obs['call']
        min_raise = obs['min_raise']
        max_raise = obs['max_raise']

        community_cards = obs['community_cards'] + pads
        hole_cards = obs['hole_cards']
        alive = list(obs['alive'])
        state_vector = alive + \
                    [obs['button'], obs['call'], obs['max_raise'], obs['min_raise'], obs['pot']] + list(obs['stacks']) + list(obs['commits']) + \
                        community_cards + hole_cards

        input_df = pd.DataFrame([state_vector], columns=[f'alive_{i}' for i in range(6)] +
                                            ['button', 'call', 'max_raise', 'min_raise', 'pot'] +
                                            [f'stack_{i}' for i in range(6)] +
                                            [f'commit_{i}' for i in range(6)] +
                                            ['board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'hole_1', 'hole_2'])


        df = input_df
        all_cards = [
            '2♠', '2♣', '2♥', '2♦', '3♠', '3♣', '3♥', '3♦', '4♠', '4♣', '4♥',
            '4♦', '5♠', '5♣', '5♥', '5♦', '6♠', '6♣', '6♥', '6♦', '7♠', '7♣',
            '7♥', '7♦', '8♠', '8♣', '8♥', '8♦', '9♠', '9♣', '9♥', '9♦', 'A♠',
            'A♣', 'A♥', 'A♦', 'J♠', 'J♣', 'J♥', 'J♦', 'K♠', 'K♣', 'K♥', 'K♦',
            'Q♠', 'Q♣', 'Q♥', 'Q♦', 'T♠', 'T♣', 'T♥', 'T♦'
        ]

        card_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        card_encoder.fit(np.array(all_cards).reshape(-1, 1))
        card_columns = [
            'board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'hole_1', 'hole_2'
        ]
        all_enc = []
        for column in card_columns:
            encoded_arr = card_encoder.transform(df[[column]].values)
            encoded_arr_df = pd.DataFrame(encoded_arr, columns=[f'{column}_{c}' for c in card_encoder.categories_[0]])
            all_enc.append(encoded_arr_df)
        all_enc_df = pd.concat(all_enc, axis=1)

        for i in range(6):
            df[f'alive_{i}'] = df[f'alive_{i}'].apply(lambda x: 1 if x is True else 0)
        df.drop(card_columns, axis=1, inplace=True)
        new_merged_arr = np.concatenate((df.values, all_enc_df.values), axis=1)
        new_merged_df = pd.DataFrame(new_merged_arr, columns=df.columns.tolist() + all_enc_df.columns.tolist())
        numerical_columns = ['call', 'max_raise', 'min_raise', 'pot'] + [f'stack_{i}' for i in range(6)] + [f'commit_{i}' for i in range(6)]
        norm = Normalizer()
        norm.fit(new_merged_df[numerical_columns].values)
        norm_new_merged_df = pd.DataFrame(norm.transform(new_merged_df[numerical_columns].values), columns=numerical_columns)
        new_merged_df.drop(numerical_columns, axis=1, inplace=True)
        new_norm_new_merged_df = pd.concat([new_merged_df, norm_new_merged_df], axis=1)

        new_norm_new_merged_df, card_encode_obsr, norm = new_norm_new_merged_df, card_encoder, norm
        vector = new_norm_new_merged_df.values
        torch_tensor = torch.tensor(vector)
        return torch_tensor.float()


    def predict(self, observation, policy_net):
        observation = self.encode(observation)
        rate = self.strategy.rate(self.current_step)
        self.current_step += 1

        if rate > random.random(): # explore
            action = random.randrange(self.num_actions) 
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(observation).argmax(dim=1).to(self.device) # exploit


def learn(agent, policy_nn, target_nn):
    batch_size = 256
    gamma = 0.999
    target_update = 10
    memory_size = 100000
    lr_decay = 0.001
    n_epochs = 3
    n_episodes = 5

    core.utils.configure()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = SequentialMemory(memory_size)
    optimizer = optim.Adam(params=policy_nn.parameters(), lr=lr_decay)

    hist = ''
    wrate = []
    stacks = []
    wins = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    total = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    print('Learning on Table 1 (DQN, MLP with 5 layers)')
    pbar = tqdm(total=n_epochs * n_episodes)
    for epoch in range(n_epochs):
        epoch_wrate = np.array([0, 0, 0, 0, 0, 0])
        for episode in range(n_episodes):
            pbar.update(1)
            env = gym.make('CustomSixPlayer-v0')
            splits = [1/3, 1/2, 3/4, 1, 2]
            players = [agents.RandomAgent(splits)] * 5 + [agents.BaseAgent()]
            env.register(players)
            obs = env.reset()
            step = 0

            while True:
                step += 1
                if obs['player'] < 5:
                    action = env.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    prev_obs = obs
                else:
                    action = agent.predict(obs, policy_nn)
                    player_id = obs['player']
                    action_type = action.value
                    if action_type == 0:
                        action_sum = -1
                    elif action_type == 5:
                        action_sum = obs['stacks'][obs['player']]
                    elif action_type == 1:
                        action_sum = obs['call']
                    else:
                        action_sum = 100

                    obs, rewards, done, info = env.step(action_sum)
                    reward = torch.from_numpy(np.array(rewards[player_id]))
                    memory.add(Experience(prev_obs, action, obs, reward))

                    if memory.ready(batch_size):
                        states, actions, rewards, next_states = memory.sample(batch_size)
                        current_q_values = get_current(policy_nn, states, actions)
                        next_q_values = get_next(target_nn, next_states)
                        target_q_values = (next_q_values * gamma) + rewards
                        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if all(done):
                    break

            if episode % target_update == 0:
                target_nn.load_state_dict(policy_nn.state_dict())

            hist += env.render() + '\n\n'
            for item in env.history:
                state, action, reward, info = item
                stacks.append(state['stacks'])
            pays = np.array(env.payouts, dtype='int')
            player = np.argmax(pays)
            epoch_wrate += pays
            wins[player] += 1
            total += pays
        epoch_wrate = np.round(epoch_wrate / n_episodes)
        wrate.append(epoch_wrate.tolist())
    pbar.close()

    wrate = np.array(wrate).T
    df_wrate = pd.DataFrame(data=wrate, dtype='int')
    df_wrate['wins'] = pd.DataFrame(wins)
    df_wrate['$/100'] = pd.DataFrame(total / (n_epochs * n_episodes)).round()
    df_wrate['bb/100'] = pd.DataFrame(total / (n_epochs * n_episodes * env.big_blind)).round()
    df_wrate['total'] = pd.DataFrame(total)
    df_wrate.to_csv('results/wrates.csv')

    with open('results/wrates.txt', 'w') as f:
        f.write(tabulate(df_wrate, tablefmt='grid', headers='keys'))

    df_stacks = pd.DataFrame(data=np.array(stacks).T, dtype='int')
    df_stacks.to_csv('results/stacks.csv')

    with open('results/stacks.txt', 'w') as f:
        f.write(tabulate(df_stacks, tablefmt='grid', headers='keys'))

    with open('results/history.txt', 'w') as f:
        f.write(hist)


if __name__ == '__main__':
    eps_start = 1
    eps_stop = 0.01
    eps_decay = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy = GreedyStrategy(eps_start, eps_stop, eps_decay)
    agent = DQNAgent(strategy, 6, device)
    policy_nn = DQNetwork(387).to(device)
    target_nn = DQNetwork(387).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())
    os.makedirs('results', exist_ok=True)
    torch.save(policy_nn.state_dict(), 'results/policy_nn.pt')
    torch.save(target_nn.state_dict(), 'results/target_nn.pt')
    learn(agent, policy_nn, target_nn)
