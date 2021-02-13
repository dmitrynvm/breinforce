import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import OneHotEncoder, Normalizer


class DQNAgent():
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, obs):

        equity_score = obs['equity_score']

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
                        community_cards + hole_cards + legal_moves + [equity_score]

        input_df = pd.DataFrame([state_vector], columns=[f'alive_{i}' for i in range(6)] +
                                            ['button', 'call', 'max_raise', 'min_raise', 'pot'] +
                                            [f'stack_{i}' for i in range(6)] +
                                            [f'commit_{i}' for i in range(6)] +
                                            ['board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'hole_1', 'hole_2'] +
                                            ['legal_fold', 'legal_call', 'legal_allin', 'raise_1/3', 'raise_1/2',
                                             'raise_3/4', 'raise_1', 'raise_3/2', 'equity_score'])

        df = input_df
        all_cards = [
            'S2', 'C2', 'H2', 'D2', 'S3', 'C3', 'H3', 'D3', 'S4', 'C4', 'H4',
            'D4', 'S5', 'C5', 'H5', 'D5', 'S6', 'C6', 'H6', 'D6', 'S7', 'C7',
            'H7', 'D7', 'S8', 'C8', 'H8', 'D8', 'S9', 'C9', 'H9', 'D9', 'SA',
            'CA', 'HA', 'DA', 'SJ', 'CJ', 'HJ', 'DJ', 'SK', 'CK', 'HK', 'DK',
            'SQ', 'CQ', 'HQ', 'DQ', 'ST', 'CT', 'HT', 'DT'
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
                                             'raise_3/4', 'raise_1', 'raise_3/2', 'equity_score']

        norm = Normalizer()
        norm.fit(df[numerical_columns].values)
        norm_new_merged_df = pd.DataFrame(norm.transform(df[numerical_columns].values),
                                          columns=numerical_columns)
        df.drop(numerical_columns, axis=1, inplace=True)

        new_norm_new_merged_df = pd.concat([df, all_enc_df, norm_new_merged_df], axis=1)

        vector = new_norm_new_merged_df.values
        torch_tensor = torch.tensor(vector)

        return torch_tensor.float()

    def predict(self, encoded_observation, policy_net):
        rate = self.strategy.rate(self.current_step)
        self.current_step += 1

        if rate > random.random(): # explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(encoded_observation).argmax(dim=1).to(self.device)