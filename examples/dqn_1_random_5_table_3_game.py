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
import plotly.graph_objects as go
from tqdm import tqdm
from time import sleep

np.random.seed(1)
pd.options.plotting.backend = "plotly"

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


def encode_data(df):
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
    return new_norm_new_merged_df, card_encoder, norm


def encode_obs(obs):
    board_cards = obs['board_cards'] + ['--' for i in range(5-len(obs['board_cards']))]
    hole_cards = obs['hole_cards']
    state_vector = [obs['player']] + list(obs['alive']) + \
                [obs['button'], obs['call'], obs['max_raise'], obs['min_raise'], obs['pot']] + list(obs['stacks']) + list(obs['commits']) + \
                    board_cards + hole_cards

    input_df = pd.DataFrame([state_vector[1:]], columns=[f'alive_{i}' for i in range(6)] +
                                        ['button', 'call', 'max_raise', 'min_raise', 'pot'] +
                                        [f'stack_{i}' for i in range(6)] +
                                        [f'commit_{i}' for i in range(6)] +
                                        ['board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'hole_1', 'hole_2'])
    new_norm_new_merged_df, card_encode_obsr, norm = encode_data(input_df)
    vector = new_norm_new_merged_df.values
    torch_tensor = torch.tensor(vector)
    return torch_tensor


def extract(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.stack(batch.reward, axis=0)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)


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

    def sample(self, n_items, extract=extract):
        return extract(random.sample(self.items, n_items))

    def ready(self, n_items):
        return n_items < len(self.items)


def get_current(policy_nn, states, actions):
    return policy_nn(states).gather(dim=1, index=actions.unsqueeze(-1))


def get_next(target_nn, next_states):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_nn(non_final_states).max(dim=1)[0].detach()
    return values


class GreedyQStrategy():
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

    def predict(self, state, policy_net):
        rate = self.strategy.rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions) # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit


def learn(agent, policy_nn, target_nn):
    batch_size = 256
    gamma = 0.999
    target_update = 10
    memory_size = 100000
    lr_decay = 0.001
    n_epochs = 100
    n_episodes = 100

    utils.configure()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = SequentialMemory(memory_size)

    optimizer = optim.Adam(params=policy_nn.parameters(), lr=lr_decay)

    hist = ''
    wrate = []
    bbpn = []
    wins = np.array([0, 0, 0, 0, 0, 0], dtype="int")
    total = np.array([0, 0, 0, 0, 0, 0], dtype="int")
    print('Learning on Table 1 (DQN, MLP with 5 layers)')
    pbar = tqdm(total=n_epochs * n_episodes)
    for epoch in range(n_epochs):
        epoch_wrate = np.array([0, 0, 0, 0, 0, 0])
        epoch_bbpn = np.array([0, 0, 0, 0, 0, 0], dtype="float")
        for episode in range(n_episodes):
            pbar.update(1)
            env = gym.make('CustomSixPlayer-v0')
            probs = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            players = [agents.BaseAgent()] * 6
            env.register(players)
            obs = env.reset()
            step = 0

            while True:
                step += 1
                enc_obs = encode_obs(obs)
                action = agent.predict(enc_obs.float(), policy_nn)
                player_id = obs['player']
                action_type = action#.numpy()[0]
                if action_type == 0:
                    action_sum = -1
                elif action_type == 5:
                    action_sum = obs['stacks'][obs['player']]
                elif action_type == 1:
                    action_sum = obs['call']
                else:
                    action_sum = 100#fracs[action_type] * obs['pot']

                obs, rewards, done, info = env.step(action_sum)
                reward = torch.from_numpy(np.array(rewards[player_id]))
                next_enc_obs = encode_obs(obs)
                memory.add(Experience(enc_obs.float(), action, next_enc_obs.float(), reward))

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

            hist += env.render() + "\n\n"
            x1 = []
            x2 = []
            x3 = []
            x4 = []
            x5 = []
            x6 = []
            for item in env.history:
                x1.append(item[0]['stacks'][0])
                x2.append(item[0]['stacks'][1])
                x3.append(item[0]['stacks'][2])
                x4.append(item[0]['stacks'][3])
                x5.append(item[0]['stacks'][4])
                x6.append(item[0]['stacks'][5])
            pays = np.array(env.payouts, dtype="int")
            player = np.argmax(pays)
            epoch_wrate += pays
            epoch_bbpn += pays / env.big_blind
            wins[player] += 1
            total += pays
        epoch_wrate = np.round(epoch_wrate / n_episodes)
        wrate.append(epoch_wrate.tolist())
        epoch_bbpn = np.round(epoch_bbpn / n_episodes)
        bbpn.append(epoch_bbpn.tolist())
    pbar.close()
    wrate = np.array(wrate)
    wrate = wrate.T
    bbpn = np.array(bbpn)
    bbpn = bbpn.T
    print(bbpn)
    episodes = np.linspace(0, n_epochs, n_epochs)

    df_wrate = pd.DataFrame(data=wrate, dtype="int")
    wrate_fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="$/100")
    ))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[0:].values[0], mode='lines+markers', name='agent_1'))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[1:].values[0], mode='lines+markers', name='agent_2'))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[2:].values[0], mode='lines+markers', name='agent_3'))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[3:].values[0], mode='lines+markers', name='agent_4'))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[4:].values[0], mode='lines+markers', name='agent_5'))
    wrate_fig.add_trace(go.Scatter(x=episodes, y=df_wrate[5:].values[0], mode='lines+markers', name='agent_6'))
    wrate_fig.write_image("results/table_1_cpne.png", width=1024, height=768)
    #wrate_fig.show()

    df_wrate["wins"] = pd.DataFrame(wins)
    df_wrate["$/100"] = df_wrate.mean(axis=1).astype("int")
    df_wrate["total"] = pd.DataFrame(total)

    str_cpne = tabulate(df_wrate, tablefmt="grid", headers="keys")
    f_cpne = open("results/table_1_cpne.txt", "w")
    f_cpne.write(str_cpne)
    f_cpne.close()

    df_bbpn = pd.DataFrame(data=bbpn, dtype="int")
    fig_bbpn = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="bb/100")
    ))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[0:].values[0], mode='lines+markers', name='agent_1'))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[1:].values[0], mode='lines+markers', name='agent_2'))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[2:].values[0], mode='lines+markers', name='agent_3'))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[3:].values[0], mode='lines+markers', name='agent_4'))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[4:].values[0], mode='lines+markers', name='agent_5'))
    fig_bbpn.add_trace(go.Scatter(x=episodes, y=df_bbpn[5:].values[0], mode='lines+markers', name='agent_6'))
    fig_bbpn.write_image("results/table_1_bbpn.png", width=1024, height=768)
    #wrate_fig.show()

    df_bbpn["wins"] = pd.DataFrame(wins)
    df_bbpn["bb/100"] = df_bbpn.mean(axis=1).astype("int")
    df_bbpn["total"] = pd.DataFrame(total)
    str_bpne = tabulate(df_bbpn, tablefmt="grid", headers="keys")

    f_bbpn = open("results/table_1_bpne.txt", "w")
    f_bbpn.write(str_bpne)
    f_bbpn.close()

    f_hist = open("results/table_1_history.txt", "w")
    f_hist.write(hist)
    f_hist.close()


if __name__ == "__main__":
    eps_start = 1
    eps_stop = 0.01
    eps_decay = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy = GreedyQStrategy(eps_start, eps_stop, eps_decay)
    agent = DQNAgent(strategy, 6, device)
    policy_nn = DQNetwork(387).to(device)
    target_nn = DQNetwork(387).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())
    #target_nn.eval()
    os.makedirs('results', exist_ok=True)
    torch.save(policy_nn.state_dict(), 'results/table_1_policy_nn_table_1.pt')
    torch.save(target_nn.state_dict(), 'results/table_1_target_nn_table_1.pt')
    learn(agent, policy_nn, target_nn)
