import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import gym
import math
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def preprocess_data_train(df):
    df = df.copy()
    all_cards = [
        '2♠', '2♣', '2♥', '2♦', '3♠', '3♣', '3♥', '3♦', '4♠', '4♣', '4♥',
        '4♦', '5♠', '5♣', '5♥', '5♦', '6♠', '6♣', '6♥', '6♦', '7♠', '7♣',
        '7♥', '7♦', '8♠', '8♣', '8♥', '8♦', '9♠', '9♣', '9♥', '9♦', 'A♠',
        'A♣', 'A♥', 'A♦', 'J♠', 'J♣', 'J♥', 'J♦', 'K♠', 'K♣', 'K♥', 'K♦',
        'Q♠', 'Q♣', 'Q♥', 'Q♦', 'T♠', 'T♣', 'T♥', 'T♦'
    ]

    card_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    card_encoder.fit(np.array(all_cards).reshape(-1, 1))
    categorical_columns_cards = ['comm_card_1', 'comm_card_2', 'comm_card_3', 'comm_card_4', 'comm_card_5', 'hole_card_1', 'hole_card_2']
    all_enc = []
    for col in categorical_columns_cards:
        encoded_arr = card_encoder.transform(df[[col]].values)
        encoded_arr_df = pd.DataFrame(encoded_arr, columns=[f'{col}_{c}' for c in card_encoder.categories_[0]])
        all_enc.append(encoded_arr_df)
    all_enc_df = pd.concat(all_enc, axis=1)

    for i in range(6):
        df[f'alive{i}'] = df[f'alive{i}'].apply(lambda x: 1 if x is True else 0)

    df.drop(categorical_columns_cards, axis=1, inplace=True) 
    new_merged_arr = np.concatenate((df.values, all_enc_df.values), axis=1)
    new_merged_df = pd.DataFrame(new_merged_arr, columns=df.columns.tolist() + all_enc_df.columns.tolist())

    numerical_columns = ['call', 'max_raise', 'min_raise', 'pot'] + [f'stack{i}' for i in range(6)] + [f'commit{i}' for i in range(6)]

    norm = Normalizer()
    norm.fit(new_merged_df[numerical_columns].values)
    norm_new_merged_df = pd.DataFrame(norm.transform(new_merged_df[numerical_columns].values), columns=numerical_columns)

    new_merged_df.drop(numerical_columns, axis=1, inplace=True)
    new_norm_new_merged_df = pd.concat([new_merged_df, norm_new_merged_df], axis=1)

    # print(new_norm_new_merged_df.shape)

    return new_norm_new_merged_df, card_encoder, norm


def preprocess_state(obs):
    # preprocess the state
    board_cards = obs['board_cards'] + ['un' for i in range(5-len(obs['board_cards']))]
    hole_cards = obs['hole_cards'][obs['player']] if len(obs['hole_cards']) > 2 else obs['hole_cards']
    state_vector = [obs['player']] + list(obs['alive']) + \
                [obs['button'], obs['call'], 
                    obs['max_raise'], obs['min_raise'], 
                    obs['pot']] + list(obs['stacks']) + list(obs['commits']) + \
                    board_cards + hole_cards

    input_df = pd.DataFrame([state_vector[1:]], columns=[f'alive{i}' for i in range(6)] + 
                                        ['button', 'call', 'max_raise', 'min_raise', 'pot'] +
                                        [f'stack{i}' for i in range(6)] +  
                                        [f'commit{i}' for i in range(6)] + 
                                        ['comm_card_1', 'comm_card_2', 'comm_card_3', 'comm_card_4', 'comm_card_5', 'hole_card_1', 'hole_card_2'])
    new_norm_new_merged_df, card_encoder, norm = preprocess_data_train(input_df)
    vector = new_norm_new_merged_df.values
    torch_tensor = torch.tensor(vector)
    return torch_tensor


class DQN(nn.Module):
    def __init__(self, input_features_size): # improve NN
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_features_size, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=6)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class ReplayMemory(): # change to Sequential
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions) # explore
            return torch.tensor([action]).to(device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(device) # exploit

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        
        non_final_state_locations = (final_state_locations == False)
        
        non_final_states = next_states[non_final_state_locations]
        
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        
        return values

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1, t2, t3, t4)

import random
from breinforce.agents.base_agent import BaseAgent


class BDQNAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def act(self, obs):
        return None

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 10000

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, 6, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(387).to(device)
target_net = DQN(387).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
import gym
from breinforce import agents, envs, utils

for episode in range(num_episodes):
        
    utils.configure()
    env = gym.make('CustomSixPlayer-v0')

    probs = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]

    game_agents = [agents.RandomAgent(probs)] * 5 + [BDQNAgent()]
    env.register(game_agents)
    obs = env.reset()

    while True:
        if obs['player'] == 5:

            state_tensor = preprocess_state(obs)
            action = agent.select_action(state_tensor.float(), policy_net)

            player_id = obs['player']

            action_type = action.numpy()[0]

            if action_type == 0:
                action_sum = -1
            elif action_type == 5:
                action_sum = obs['stacks'][obs['player']]
            elif action_type == 1:
                action_sum = obs['call']
            else:
                action_sum = 100#fracs[action_type] * obs['pot']
        
            # print(action_sum)
            hand = env.step(action_sum)

            obs, rewards, done, info = hand
            # print(rewards)

            reward = torch.from_numpy(np.array([rewards[player_id]]))

            next_state_tensor = preprocess_state(obs)

            memory.push(Experience(state_tensor.float(), action, next_state_tensor.float(), reward))

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)
                
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            action = env.act(obs)
            hand = env.step(action)

            obs, rewards, done, info = hand

        if all(done):
            break
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print('Episode: ', episode)