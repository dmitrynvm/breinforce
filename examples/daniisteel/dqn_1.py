import gym
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from breinforce import agents, core, envs
from tabulate import tabulate
from tqdm import tqdm
from utils import get_score, flatten, legal_actions, get_legal_action, get_current, get_next
from memories import SequentialMemory
from strategies import GreedyStrategy
from networks import MLPNetwork
from rl_agents import DQNAgent


np.random.seed(1)
pd.options.plotting.backend = 'plotly'

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
Action = namedtuple('Action', ['name', 'value'])


def learn(agent, policy_nn, target_nn, memory, hyperparameters):
    batch_size = hyperparameters['batch_size']
    gamma = hyperparameters['gamma']
    target_update = hyperparameters['target_update']
    lr_decay = hyperparameters['lr_decay']
    n_epochs = hyperparameters['n_epochs']
    n_episodes = hyperparameters['n_episodes']

    core.utils.configure()

    optimizer = optim.Adam(params=policy_nn.parameters(), lr=lr_decay)

    hist = ''
    wrate = []
    stacks = []
    wins = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    total = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    print('Learning on Table 1 (DQN, MLP with 5 layers)')
    pbar = tqdm(total=n_epochs * n_episodes)
    count_episodes = 0
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
            count_episodes += 1
            while True:
                step += 1
                if obs['player'] < 5:
                    action = env.predict(obs)
                    obs, rewards, done = env.step(action)

                else:
                    # adding legal actions from valid actions to obs
                    l_actions = legal_actions(obs, splits)
                    l_actions = flatten(l_actions, parent_key='', sep='_')
                    obs['legal_actions'] = l_actions

                    # adding judge score to obs
                    equity_score = get_score(obs)
                    obs['equity_score'] = equity_score

                    # encode obs
                    encoded_obs = agent.encode(obs)

                    # get prediction
                    action = agent.predict(encoded_obs, policy_nn)

                    player_id = obs['player']

                    a_w_types = {0: 'fold', 1: 'raise_1/3', 2: 'raise_1/2', 3: 'raise_3/4',
                                 4: 'raise_1', 5: 'raise_3/2', 6: 'call', 7: 'allin'}

                    # post processing get legal action from prediction
                    action, action_sum = get_legal_action(obs, action, a_w_types)

                    the_action = Action(a_w_types[action.numpy()[0]], action_sum)
                    obs, rewards, done = env.step(the_action)

                    # adding legal actions and equity score to new obs and encoding new obs
                    l_actions = legal_actions(obs, splits)
                    l_actions = flatten(l_actions, parent_key='', sep='_')
                    obs['legal_actions'] = l_actions
                    obs['equity_score'] = get_score(obs)
                    new_encoded_obs = agent.encode(obs)

                    reward = torch.from_numpy(np.array(rewards[player_id]))
                    memory.add(Experience(encoded_obs.float(), action, new_encoded_obs.float(), reward))

                    if memory.ready(batch_size):
                        states, actions, rewards, next_states = memory.sample(batch_size)
                        current_q_values = get_current(policy_nn, states, actions)
                        next_q_values = get_next(target_nn, next_states)
                        target_q_values = (next_q_values * gamma) + rewards
                        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss = loss

                if all(done):
                    break

            if count_episodes % target_update == 0:
                print('target network update happened')
                target_nn.load_state_dict(policy_nn.state_dict())

            hist += env.render() + '\n\n'
            for item in env.history:
                state, action, reward = item
                stacks.append(state['stacks'])
            pays = np.array(env.state.rewards, dtype='int')
            player = np.argmax(pays)
            epoch_wrate += pays
            wins[player] += 1
            total += pays

        if epoch % 50 == 1:
            # saving checkpoint
            try:
                e = epoch_loss.item()
            except:
                e = 'nan'
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoints/latest_model_{epoch}_{e}_newarch_tau2.pt')

        epoch_wrate = np.round(epoch_wrate / n_episodes)
        wrate.append(epoch_wrate.tolist())
    pbar.close()

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


if __name__ == '__main__':

    hyperparameters = dict()
    hyperparameters['eps_start'] = 1
    hyperparameters['eps_stop'] = 0.01
    hyperparameters['eps_decay'] = 0.001

    hyperparameters['batch_size'] = 256
    hyperparameters['gamma'] = 0.999
    hyperparameters['target_update'] = 1000
    hyperparameters['memory_size'] = 100000
    hyperparameters['lr_decay'] = 0.001
    hyperparameters['n_epochs'] = 5
    hyperparameters['n_episodes'] = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy = GreedyStrategy(hyperparameters['eps_start'], hyperparameters['eps_stop'], hyperparameters['eps_decay'])
    memory = SequentialMemory(hyperparameters['memory_size'])
    agent = DQNAgent(strategy, 8)
    policy_nn = MLPNetwork(396, 8).to(device)
    target_nn = MLPNetwork(396, 8).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    learn(agent, policy_nn, target_nn, memory, hyperparameters)
