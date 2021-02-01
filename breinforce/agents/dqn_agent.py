import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from breinforce.agents import SplitAgent


class DQNAgent(SplitAgent):
    def __init__(self, splits, strategy, num_actions, device):
        super().__init__(splits)
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.legal_actions = []

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

    def predict(self, obs, policy_net):
        action = None
        encoded = self.encode(obs)
        rate = self.strategy.rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions) 
            action = torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                action = policy_net(encoded).argmax(dim=1).to(self.device)
        action = action.tolist()[0]

        splits = [0.3, 0.5, 0.75, 1, 2]
        legal_actions = get_legal_actions(
            obs['call'],
            obs['min_raise'],
            obs['max_raise'],
            obs['pot'],
            splits
        )
        return self.clean(action, legal_actions)

    def clean(self, action, legal_actions) -> int:
        '''
        Find closest bet size to actual bet
        '''
        index = np.argmin(np.absolute(np.array(list(legal_actions.values())) - action))
        return list(legal_actions.items())[index]
