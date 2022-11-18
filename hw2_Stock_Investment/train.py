import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions 

#%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
#from tqdm.notebook import tqdm
from tqdm import tqdm

import os
import pandas as pd
import json

# 套stable_baselines3模型
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN, PPO

# 設定 seed
torch.manual_seed(1234)
np.random.seed(1234)

'''做完ㄉ
1-----------------------------------
#改reward
    model:只用'Close','Open', reward是rewards.append(np.full(total_step, info['total_reward']))
    model_all_feature:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是rewards.append(np.full(total_step, info['total_reward']))
    model_single_reward:只用'Close','Open', reward是每個動作自己產生的reward
    model_all_feature_single_reward:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是每個動作自己產生的reward
        # Suitable Credit (gamma)
    model_gamma_reward:只用'Close','Open', reward是gamma_reward
    model_all_feature_gamma_reward:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是gamma_reward
    model_all_feature_gamma_reward_2:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是gamma_reward
        # Baseline
    model_all_feature_baseline_p1:                  reward全是total_reward-0.1
    model_all_feature_gamma_reward_baseline_p1:     reward是gamma_reward-0.1
    model_all_feature_baseline_p1634:               reward全是total_reward-0.1634
    model_all_feature_gamma_reward_baseline_p1634:  reward是gamma_reward-0.1634
    model_all_feature_baseline_1:                   reward全是total_reward-1
    model_all_feature_gamma_reward_baseline_1:      reward是gamma_reward-1
        # band reward
    model_band
    model_band_10times
2-----------------------------------
#改神經網路大小(original->bigNN, smallNN)(選用model、model_all_feature、model_band_10times)
    q2_model_bigNN:只用'Close','Open', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_smallNN:只用'Close','Open', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_all_feature_bigNN:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_all_feature_smallNN:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_band_10times_bigNN
    q2_model_band_10times_smallNN
#改一個 batch 中的回合數(5->1, 10)(選用model、model_all_feature、model_band_10times)
    q2_model_EPISODEPERBATCH_1:只用'Close','Open', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_EPISODEPERBATCH_10:只用'Close','Open', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_all_feature_EPISODEPERBATCH_1:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_all_feature_EPISODEPERBATCH_10:用'Shares','Amount','Open','High','Low','Close','Change','Turnover', reward是rewards.append(np.full(total_step, info['total_reward']))
    q2_model_band_10times_EPISODEPERBATCH_1
    q2_model_band_10times_EPISODEPERBATCH_10
#改frame_bound((1000, 1500)->(1500, 2000), (1000, 1100))(換資料區間跟縮小資料區間)(避開測試的(500, 1000)以及(2000, 2500))
    q2_model_change
    q2_model_minify
    q2_model_all_feature_change
    q2_model_all_feature_minify
    q2_model_band_10times_change
    q2_model_band_10times_minify
'''
def main():
    STOCKS_TSMC = pd.read_csv(os.path.join('.', 'hw2_Stock_Investment', 'CE6020_hw2_resource', '2330_stock.csv'))
    print(STOCKS_TSMC.head)
    
    #### 環境跟變數處理
    # 將台積電資料輸入股票環境，並設定本次環境範圍與輸入天數資料 (欄位內容、天數等)。
    def my_process_data(env):
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
        # 這邊可自訂想要使用的 feature: 'Shares','Amount','Open','High','Low','Close','Change','Turnover'
        # 原來: 'Close', 'Open'
        signal_features = env.df.loc[:, ['Close', 'Open']].to_numpy()[start:end]
        return prices, signal_features
    class MyStocksEnv(StocksEnv):
        _process_data = my_process_data
    # window_size: 能夠看到幾天的資料當作輸入, frame_bound: 想要使用的資料日期區間
    # 可修改 frame_bound 來學習不同的環境資料
    # 不可修改此處 window_size 參數
    env = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(1000, 1500)) # trade_fee_bid_percent賣股票手續費 trade_fee_ask_percent買股票手續費
    # 檢視環境參數
    print("env information:")
    print("> shape:", env.shape)
    print("> df.shape:", env.df.shape)
    print("> prices.shape:", env.prices.shape)
    print("> signal_features.shape:", env.signal_features.shape)
    print("> max_possible_profit:", env.max_possible_profit())
    env.reset()
    #env.render()
    #plt.show()
    
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(info)
            break
    plt.cla()
    #env.render_all()
    #plt.show()
    
    
    ##### 定義PolicyGradientNetwork跟PolicyGradientAgent
    # Policy Gradient, 我們預設模型的輸入是 20-dim (10天*2欄位) 的 observation，輸出則是離散的二個動作之一 (賣=0 或 買=1)
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
            # original
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
            
    '''
    再來，搭建一個簡單的 agent，並搭配上方的 policy network 來採取行動。 這個 agent 能做到以下幾件事：
    learn()：從記下來的 log probabilities 及 rewards 來更新 policy network。
    sample()：從 environment 得到 observation 之後，利用 policy network 得出應該採取的行動。 而此函式除了回傳抽樣出來的 action，也會回傳此次抽樣的 log probabilities。
    '''
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
    
    
    
    
    #### 實際訓練
    # 建立一個 network 和 agent以進行訓練。
    network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
    agent = PolicyGradientAgent(network)
    
    #訓練 Agent
    #現在我們開始訓練 agent。 透過讓 agent 和 environment 互動，我們記住每一組對應的 log probabilities 及 reward，並在資料日期區間結束後，回放這些「記憶」來訓練 policy network。
    EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent --> 回合數(1,5,10)
    NUM_BATCH = 400     # 總共更新 400 次
    CHECKPOINT_PATH = os.path.join('.', 'hw2_Stock_Investment', 'q2_model_smallNN.ckpt') # agent model 儲存位置
    avg_total_rewards = []
    agent.network.train()  # 訓練前，先確保 network 處在 training 模式
    
    prg_bar = tqdm(range(NUM_BATCH))
    
    for batch in prg_bar:
        log_probs, rewards = [], []
        total_rewards = []
        # 蒐集訓練資料
        for episode in range(EPISODE_PER_BATCH):  
            observation = env.reset()
            total_step = 0
            
            # 自創，step_rewards用以紀錄每次迴圈的每個reward
            step_rewards = []
            
            while True:
                action, log_prob = agent.sample(observation)
                observation, reward, done, info = env.step(action)
                
                #print('-----> observation, reward, done, info', observation, reward, done, info)
                
                # just single reward
                #step_rewards.append(reward)
                # baseline reward
                #baseline = 0.1634 #0.1, 0.1634, 1, 10
                #step_rewards.append(reward - baseline)
                #print('--->', reward, type(reward))

                log_probs.append(log_prob)
                total_step += 1
                if done:
                    total_rewards.append(info['total_reward'])
                    # gamma reward
                    
                    #gamma = 0.9
                    #reward_arr = np.array(step_rewards)
                    #for i in range(len(reward_arr) - 2, -1, -1):
                    #    # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
                    #    reward_arr[i] += gamma * reward_arr[i + 1]
                    # normalize episode rewards
                    #reward_arr -= np.mean(reward_arr)
                    #reward_arr /= np.std(reward_arr)
                    #rewards.append(reward_arr)
                    
                    
                    # single_rewards
                    #rewards.append(np.array(step_rewards))
                    
                    # total_reward as rewards
                    rewards.append(np.full(total_step, info['total_reward']))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                    
                    #print('--------->', rewards[-1].shape, rewards[-1])
                    break
        # 紀錄訓練過程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_total_rewards.append(avg_total_reward)
        prg_bar.set_description(f"Average Reward: {avg_total_reward: 4.2f}, Final Reward: {info['total_reward']: 4.2f}, Final Profit: {info['total_profit']: 4.2f}")
        # 更新網路
        rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    
    '''
    # 自創波段reward算法
    for batch in prg_bar:
        log_probs, rewards = [], []
        total_rewards = []
        # 蒐集訓練資料
        for episode in range(EPISODE_PER_BATCH):  
            observation = env.reset()
            total_step = 0
            
            # 自創，step_rewards用以紀錄每次迴圈的每個reward，band_rewards用以紀錄波段reward
            step_rewards = []
            band_rewards = []
            # 波段起點終點
            start_idx, end_idx = -1, -1
            while True:
                action, log_prob = agent.sample(observation)
                observation, reward, done, info = env.step(action)
                
                #print('-----> reward, done, info', reward, done, info)
                #print(len(step_rewards), start_idx, end_idx, len(band_rewards))
                if info['position'] == 0:
                    # 無持股
                    if start_idx == end_idx:
                        start_idx += 1
                        end_idx += 1
                        band_rewards.append(0)
                    # 波段獲利計算(後-前再除上所需時間)
                    elif end_idx > start_idx:
                        band_reward = 10 * sum(step_rewards[start_idx:end_idx]) / (end_idx-start_idx)
                        for band_i in range(end_idx-start_idx+1):
                            band_rewards.append(band_reward)
                        start_idx = end_idx
                # 持股
                elif info['position'] == 1:
                    end_idx += 1
                #print(len(step_rewards), start_idx, end_idx, len(band_rewards))
                #print('-----> start_idx, end_idx, band_rewards', start_idx, end_idx, band_rewards)
            
                
                # step_rewards
                step_rewards.append(reward)
                log_probs.append(log_prob)
                total_step += 1
                if done:
                    # 最後要做一次(不能保證結尾是0可以獲利了結)
                    if len(band_rewards) != len(step_rewards):
                        #print('-------------------')
                        #print('len(band_rewards)', len(band_rewards))
                        #print('len(step_rewards)', len(step_rewards))
                        #print('start_idx, end_idx', start_idx, end_idx)
                        
                        band_reward = 10 * sum(step_rewards[start_idx:end_idx]) / (end_idx-start_idx)
                        for band_i in range(end_idx-start_idx):
                            band_rewards.append(band_reward)
                        start_idx = end_idx
                        
                        #print('len(band_rewards)', len(band_rewards))
                        #print('len(step_rewards)', len(step_rewards))
                        #print('start_idx, end_idx', start_idx, end_idx)
                    
                    
                    total_rewards.append(info['total_reward'])
                    rewards.append(np.array(band_rewards))
                    
                    #print('--------->', rewards[-1].shape, rewards[-1])
                    break
        # 紀錄訓練過程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_total_rewards.append(avg_total_reward)
        prg_bar.set_description(f"Average Reward: {avg_total_reward: 4.2f}, Final Reward: {info['total_reward']: 4.2f}, Final Profit: {info['total_profit']: 4.2f}")
        # 更新網路
        rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        '''
    # 儲存 agent model 參數
    agent.save_ckpt(CHECKPOINT_PATH)
    
    
    #### 訓練後
    #訓練結果
    #訓練過程中，我們持續記下了 avg_total_reward，這個數值代表的是：每次更新 policy network 前，我們讓 agent 玩數個回合（episodes），而這些回合的平均 total rewards 為何。 理論上，若是 agent 一直在進步，則所得到的 avg_total_reward 也會持續上升。 若將其畫出來則結果如下：
    plt.plot(avg_total_rewards)
    plt.title("Total Rewards")
    plt.show()
    
    #測試
    #在這邊我們替換環境使用的資料日期區間，並使用讀取紀錄點的方式來執行測試。
    env = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(2000, 2500))
    network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
    test_agent = PolicyGradientAgent(network)
    checkpoint_path = os.path.join('.', 'hw2_Stock_Investment', 'q2_model_all_feature_bigNN.ckpt')
    test_agent.load_ckpt(checkpoint_path)
    test_agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

    observation = env.reset()
    while True:
        action, _ = test_agent.sample(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
    plt.cla()
    env.render_all()
    plt.show()
    


'''
3-----------------------------------
確認A2C_main的reward是對的
用A2C,DQN,PPO/用全feature,2feature/用MlpPolicy  ----->找最佳
A2C_MlpPolicy_100000
A2C_MlpPolicy_100000_all_feature
DQN_MlpPolicy_100000
DQN_MlpPolicy_100000_all_feature
PPO_MlpPolicy_100000
PPO_MlpPolicy_100000_all_feature

用最佳模型加入from finta import TA的趨勢線等等
'''   
def stable_baselines3_main():
    STOCKS_TSMC = pd.read_csv(os.path.join('.', 'hw2_Stock_Investment', 'CE6020_hw2_resource', '2330_stock.csv'))
    print(STOCKS_TSMC.head)
    
    #### 環境跟變數處理
    # 將台積電資料輸入股票環境，並設定本次環境範圍與輸入天數資料 (欄位內容、天數等)。
    def my_process_data(env):
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
        # 這邊可自訂想要使用的 feature: 'Shares','Amount','Open','High','Low','Close','Change','Turnover'
        # 原來: 'Close', 'Open'
        signal_features = env.df.loc[:, ['Shares','Amount','Open','High','Low','Close','Change','Turnover']].to_numpy()[start:end]
        return prices, signal_features
    class MyStocksEnv(StocksEnv):
        _process_data = my_process_data
    # window_size: 能夠看到幾天的資料當作輸入, frame_bound: 想要使用的資料日期區間
    # 可修改 frame_bound 來學習不同的環境資料
    # 不可修改此處 window_size 參數
    env = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(1000, 1500)) # trade_fee_bid_percent賣股票手續費 trade_fee_ask_percent買股票手續費
    # 檢視環境參數
    print("env information:")
    print("> shape:", env.shape)
    print("> df.shape:", env.df.shape)
    print("> prices.shape:", env.prices.shape)
    print("> signal_features.shape:", env.signal_features.shape)
    print("> max_possible_profit:", env.max_possible_profit())
    env.reset()
    #env.render()
    #plt.show()
    
    
    # training
    model = DQN('MlpPolicy', env, verbose=1) #A2C, DQN, PPO
    model.learn(total_timesteps=100000)
    model.save("DQN_MlpPolicy_100000")
    
    # Evaluation:
    env_eva = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(2000, 2500))
    obs = env_eva.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_eva.step(action)
        if done:
            print("info", info)
            break
    plt.figure(figsize=(15,6))
    plt.cla()
    env_eva.render_all()
    plt.show()
    
    # testing
    env_eva = MyStocksEnv(df=STOCKS_TSMC, window_size=10, frame_bound=(2000, 2500))
    model = DQN.load("DQN_MlpPolicy_100000_all_feature")
    
    actions_list = []
    total_reward = 0
    
    obs = env_eva.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_eva.step(action)
        #print('action, rewards, dones, info', action, rewards, dones, info)
        actions_list.append(action)
        total_reward += rewards
        if dones:
            break
        #env_eva.render()
    plt.cla()
    env_eva.render_all()
    plt.show()
    
    
    # 輸出檔案內容必須包含以下五個資料
    output_content = {
        'start': 2000,
        'end': 2500,
        'total_reward': info['total_reward'],
        'total_profit': info['total_profit'],
        'action': list(map(int, actions_list))# actions_list
    }

    # 輸出模型動作選擇文件
    with open('output_DQN_MlpPolicy_100000_all_feature.txt', 'w+') as fs:
        json.dump(output_content, fs)

    
    
    
    '''
    #from finta import TA


    env = gym.make('stocks-v0', df=df, frame_bound=(5,250), window_size=5)
    env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,50))

    # Build Environment and Train:
    env_maker = lambda: env2
    env = DummyVecEnv([env_maker])

    model = A2C('MlpLstmPolicy', env, verbose=1) 
    model.learn(total_timesteps=1000000)

    # Evaluation:
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(80,250))
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()
    '''


if __name__ == '__main__':
    main()
    #stable_baselines3_main()