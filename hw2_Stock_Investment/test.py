import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import os
import argparse
import pandas as pd
import json

"""輸入參數設計 (不可動)
"""
parser = argparse.ArgumentParser()
parser.add_argument('--start', '-s', type=int, default=10,
                    help='stock investment start date index')
parser.add_argument('--end', '-e', type=int, default=300,
                    help='stock investment end date index')
parser.add_argument('--output_action_file_pathname', '-f', type=str, default='action.txt',
                    help='output action file pathname')

args = parser.parse_args()

predict_start_date_index = args.start
predict_end_date_index = args.end
output_action_file_pathname = args.output_action_file_pathname

""" 以下參數之設定僅提供 ipynb 檔案測試用 (不得用於繳交之 test.py 檔案)
predict_start_date_index = 10, predict_end_date_index = 300, output_action_file_pathname = 'action.txt'
"""
# predict_start_date_index = 10
# predict_end_date_index = 300
# output_action_file_pathname = 'action.txt'

STOCKS_TSMC = pd.read_csv(os.path.join('.', 'hw2_Stock_Investment', 'CE6020_hw2_resource', '2330_stock.csv'))

"""模型設計 (可動)
network(PolicyGradientNetwork), optimizer 設計可依照同學設計做修改
"""
# 模型記錄點檔案位置
checkpoint_file_path = os.path.join('.', 'hw2_Stock_Investment', 'q2_model_smallNN.ckpt')

# 設定 seed
torch.manual_seed(1234)
np.random.seed(1234)

# 請把您的模型複製過來
class PolicyGradientNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # original
        '''
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        '''
        # big
        '''
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)
        '''
        #small
        
        self.fc1 = nn.Linear(input_dim, 4)
        self.fc2 = nn.Linear(4, 2)
        
    def forward(self, state):
        '''
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
        '''
        # big
        '''
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        hid = torch.tanh(self.fc3(hid))
        hid = torch.tanh(self.fc4(hid))
        hid = torch.tanh(self.fc5(hid))
        return F.softmax(self.fc6(hid), dim=-1)
        '''
        #small
        
        hid = torch.tanh(self.fc1(state))
        return F.softmax(self.fc2(hid), dim=-1)
        
class PolicyGradientAgent():
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def sample(self, state):
        state = state.flatten()
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    def load_ckpt(self, ckpt_path):
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Checkpoint file not found, use default settings")
    def save_ckpt(self, ckpt_path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)

"""模型參數設計 (可動)
使用之資料欄位可依照同學設計做修改
"""

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    # 這邊可自訂想要使用的 feature
    signal_features = env.df.loc[:, ['Close', 'Open']].to_numpy()[start:end]
    return prices, signal_features
class MyStocksEnv(StocksEnv):
    _process_data = my_process_data
# window_size: 能夠看到幾天的資料當作輸入, frame_bound: 想要使用的資料區間
env = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(predict_start_date_index, predict_end_date_index))

"""測試 Agent 模型預測結果"""

network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
test_agent = PolicyGradientAgent(network)

test_agent.load_ckpt(checkpoint_file_path)
test_agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

state = env.reset()

actions_list = []
total_reward = 0

while True:
    action, _ = test_agent.sample(state)
    state, reward, done, info = env.step(action)
    actions_list.append(action)

    total_reward += reward
    if done:
        break

# 輸出檔案內容必須包含以下五個資料
output_content = {
    'start': predict_start_date_index,
    'end': predict_end_date_index,
    'total_reward': info['total_reward'],
    'total_profit': info['total_profit'],
    'action': actions_list
}

# 輸出模型動作選擇文件
with open(output_action_file_pathname, 'w+') as fs:
    json.dump(output_content, fs)

#python ./hw2_Stock_Investment/test.py –s 500 –e 1000 –f output_q2_model_smallNN.txt