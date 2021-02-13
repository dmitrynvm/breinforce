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
from tqdm import tqdm
from time import sleep
from fractions import Fraction
import collections

core.utils.configure()

np.random.seed(1)

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
Action = namedtuple('Action', ['name', 'value'])


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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


class MLPNetwork(nn.Module):
    def __init__(self, n_features, n_output):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=100)
        self.fc5 = nn.Linear(in_features=100, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=n_output)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return x


def greedy_rate(step, eps_start, eps_stop, eps_decay):
    return eps_stop + (eps_start - eps_stop) * math.exp(-1. * step * eps_decay)


class DQNAgent():

    def __init__(
        self,
        num_actions,
        eps_start=1,
        eps_stop=0.01,
        eps_decay=0.001
    ):
        self.current_step = 0
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_stop = eps_stop
        self.eps_decay = eps_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, obs):
        possible_moves = ['fold', 'call', 'allin', 'raise_1/3', 'raise_1/2', 'raise_3/4', 'raise_1', 'raise_3/2']
        legal_moves = []
        for move in possible_moves:
            if move in obs['legal_actions']:
                move_sum = obs['legal_actions'][move]
            else:
                move_sum = -10
            legal_moves.append(move_sum)

        pads = ['--' for i in range(5 - len(obs['community_cards']))]
        community_cards = obs['community_cards'] + pads
        hole_cards = obs['hole_cards']
        alive = list(obs['alive'])

        if len(obs['valid_actions']) == 0:
            valid_call = -10
            raise_min = -10
            raise_max = -10
        else:
            valid_call = obs['valid_actions']['call']
            raise_min = obs['valid_actions']['raise']['min']
            raise_max = obs['valid_actions']['raise']['max']

        state_vector = alive + \
                    [obs['button'], valid_call, raise_min,
                     raise_max, obs['pot']] + list(obs['stacks']) + list(obs['commits']) + \
                        community_cards + hole_cards + legal_moves

        input_df = pd.DataFrame([state_vector], columns=[f'alive_{i}' for i in range(6)] +
                                            ['button', 'call', 'max_raise', 'min_raise', 'pot'] +
                                            [f'stack_{i}' for i in range(6)] +
                                            [f'commit_{i}' for i in range(6)] +
                                            ['board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'hole_1', 'hole_2'] +
                                            ['legal_fold', 'legal_call', 'legal_allin', 'raise_1/3', 'raise_1/2',
                                             'raise_3/4', 'raise_1', 'raise_3/2'])

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

        numerical_columns = ['call', 'max_raise', 'min_raise', 'pot'] + [f'stack_{i}' for i in range(6)] \
                            + [f'commit_{i}' for i in range(6)] + ['legal_fold', 'legal_call', 'legal_allin', 'raise_1/3', 'raise_1/2',
                                             'raise_3/4', 'raise_1', 'raise_3/2']

        norm = Normalizer()
        norm.fit(df[numerical_columns].values)
        norm_new_merged_df = pd.DataFrame(norm.transform(df[numerical_columns].values),
                                          columns=numerical_columns)
        df.drop(numerical_columns, axis=1, inplace=True)

        new_norm_new_merged_df = pd.concat([df, all_enc_df, norm_new_merged_df], axis=1)

        vector = new_norm_new_merged_df.values
        torch_tensor = torch.tensor(vector)

        return torch_tensor.float()

    def predict(self, obs, policy):
        obs = self.encode(obs)
        rate = greedy_rate(
            self.current_step,
            self.eps_start,
            self.eps_stop,
            self.eps_decay
        )
        self.current_step += 1

        if rate > random.random(): # explore
            action = random.randrange(self.num_actions) 
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy(obs).argmax(dim=1).to(self.device) # exploit


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
    return out


def learn(agent, policy_net, target_net):
    batch_size = 256
    gamma = 0.999
    target_update = 10
    memory_size = 100000
    lr_decay = 0.001
    n_epochs = 5
    n_episodes = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = SequentialMemory(memory_size)
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr_decay)

    hist = ''
    wrate = []
    stacks = []
    wins = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    total = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    print('Learning on Table 1 (DQN, MLP with 5 layers)')
    pbar = tqdm(total=n_epochs * n_episodes)
    agent_out = ''
    for epoch in range(n_epochs):
        epoch_wrate = np.array([0, 0, 0, 0, 0, 0])
        epoch_loss = 0
        for episode in range(n_episodes):
            pbar.update(1)
            env = gym.make('CustomSixPlayer-v0')
            splits = [1/3, 1/2, 3/4, 1, 3/2]
            players = [agents.RandomAgent(splits)] * 5 + [agents.BaseAgent()]
            env.register(players)
            obs = env.reset()
            step = 0

            while True:
                step += 1
                if obs['player'] < 5:
                    action = env.predict(obs)
                    obs, rewards, done = env.step(action)
                else:
                    l_actions = legal_actions(obs, splits)
                    l_actions = flatten(l_actions, parent_key='', sep='_')
                    obs['legal_actions'] = l_actions

                    action = agent.predict(obs, policy_net)
                    player_id = obs['player']
                    action_type = action.numpy()[0]

                    a_w_types = {0: 'fold', 1: 'raise_1/3', 2: 'raise_1/2', 3: 'raise_3/4',
                                 4: 'raise_1', 5: 'raise_3/2', 6: 'call', 7: 'allin'}

                    if a_w_types[action.numpy()[0]] not in obs['legal_actions']:
                        if action_type == 1 or action_type == 2 or action_type == 3 or action_type == 4:
                            action = torch.tensor([0])
                            action_value = obs['legal_actions']['fold']
                        else:
                            action = torch.tensor([7])
                            action_value = obs['legal_actions']['allin']
                    else:
                        if action_type == 0:
                            action_value = obs['legal_actions']['fold']
                        elif action_type == 7:
                            action_value = obs['legal_actions']['allin']
                        elif action_type == 6:
                            action_value = obs['legal_actions']['call']
                        elif action_type == 1:
                            action_value = obs['legal_actions']['raise_1/3']
                        elif action_type == 2:
                            action_value = obs['legal_actions']['raise_1/2']
                        elif action_type == 3:
                            action_value = obs['legal_actions']['raise_3/4']
                        elif action_type == 4:
                            action_value = obs['legal_actions']['raise_1']
                        else:
                            action_value = obs['legal_actions']['raise_3/2']

                    prev_obs = obs

                    the_action = Action(a_w_types[action_type], action_value)

                    policy_out = str(obs) + '-> '
                    obs, rewards, done = env.step(the_action)
                    print(the_action, rewards)
                    policy_out += str(the_action) + '\n'
                    agent_out += policy_out

                    l_actions = legal_actions(obs, splits)
                    l_actions = flatten(l_actions, parent_key='', sep='_')
                    obs['legal_actions'] = l_actions
                    encoded_obs = agent.encode(obs)
                    prev_encoded_obs = agent.encode(prev_obs)

                    reward = torch.from_numpy(np.array(rewards[player_id]))
                    memory.add(Experience(prev_encoded_obs.float(), action, encoded_obs.float(), reward))

                    if memory.ready(batch_size):
                        states, actions, rewards, next_states = memory.sample(batch_size)
                        current_q_values = get_current(policy_net, states, actions)
                        next_q_values = get_next(target_net, next_states)
                        target_q_values = (next_q_values * gamma) + rewards
                        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss = loss

                if all(done):
                    break

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            hist += env.render() + '\n\n'
            for item in env.history:
                state, action, reward = item
                stacks.append(state['stacks'])
            pays = np.array(env.state.rewards, dtype='int')
            player = np.argmax(pays)
            epoch_wrate += pays
            wins[player] += 1
            total += pays

        '''
        if epoch % 1000 == 0:
            # saving checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, 'checkpoints/latest_model.pt')
        '''

        epoch_wrate = np.round(epoch_wrate / n_episodes)
        wrate.append(epoch_wrate.tolist())
    pbar.close()

    os.makedirs('results', exist_ok=True)
    wrate = np.array(wrate).T
    df_wrate = pd.DataFrame(data=wrate, dtype='int')
    df_wrate['wins'] = pd.DataFrame(wins)
    df_wrate['$/100'] = pd.DataFrame(total / (n_epochs * n_episodes)).round()
    df_wrate['bb/100'] = pd.DataFrame(total / (n_epochs * n_episodes * env.state.big_blind)).round()
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

    with open('results/agent_out.txt', 'w') as f:
        f.write(agent_out)


if __name__ == '__main__':
    agent = DQNAgent(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = MLPNetwork(395, 8).to(device)
    target_net = MLPNetwork(395, 8).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    learn(agent, policy_net, target_net)
