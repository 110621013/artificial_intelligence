
'''
不同模型比較(linear dnn LSTM) -> 完成, linear勝出, dnn跟LSTM以及有reLU的linear都會有強烈高估
不同Learning Rate
 比較取前2 天和前4 天的資料
 比較只取部分特徵和取所有特徵的情況下
 比較資料在有無Normalization
請說明你超越Baseline 的Model是如何實作的。 (對股市K棒多理解並做出新feather)
'''

#install h5py==2.10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.optim import optimizer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, BatchNormalization, MaxPooling1D, TimeDistributed #, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping#, ModelCheckpoint

def get_trainx_trainy(input_date_data_size, remove_columns_list=[], nor_flag=True):
    ## Load Data, 處理掉不需要的欄位
    train_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','train.csv'))
    train_df.drop(columns=['datetime'], inplace=True)
    train_df_origin = train_df.copy()
    
    ## Preprocess 以前四天的資料預測第五天的收盤(欄位 'close')結果
    # 選擇要預測的欄位順序
    output_predict_index = train_df_origin.columns.get_loc('close')
    
    for remove_columns_name in remove_columns_list:
        train_df.drop(columns=[remove_columns_name], inplace=True)
    
    # 設定 seed
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    train = train_df.to_numpy()
    train_origin = train_df_origin.to_numpy()
    train_size, feature_size = train.shape
    # 以一段時間的資料當作輸入，故資料數量要扣掉輸入天數範圍
    train_size = train_size - input_date_data_size
    print('train_size, feature_size:', train_size, feature_size)
    
    train_x = np.empty([train_size, feature_size * input_date_data_size], dtype = float)
    train_y = np.empty([train_size, 1], dtype = float)

    for idx in range(train_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, train[idx + count]])
        train_x[idx, :] = temp_data
        train_y[idx, 0] = train_origin[idx + input_date_data_size][output_predict_index]
    print('train_x.shape, train_y.shape:', train_x.shape, train_y.shape)
    
    ## Standardize
    if nor_flag:
        mean_x = np.mean(train_x, axis = 0)
        std_x = np.std(train_x, axis = 0)
        for i in range(len(train_x)):
            for j in range(len(train_x[0])):
                if std_x[j] != 0:
                    train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]
    else:
        mean_x, std_x = False, False
    
    ## Training
    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    train_y = train_y.view(train_y.shape[0], 1)
    
    val_rate = 0.1
    val_x_seq, val_y_seq = train_x[int(train_x.shape[0]*(1-val_rate)):], train_y[int(train_y.shape[0]*(1-val_rate)):]
    
    ################
    ## shuffle
    randomList = np.arange(train_x.shape[0])
    np.random.shuffle(randomList)
    train_x_shuffle, train_y_shuffle = train_x[randomList], train_y[randomList]
    
    ## create Val
    train_x = train_x_shuffle[int(train_x_shuffle.shape[0]*val_rate):]
    train_y = train_y_shuffle[int(train_y_shuffle.shape[0]*val_rate):]
    val_x = train_x_shuffle[:int(train_x_shuffle.shape[0]*val_rate)]
    val_y = train_y_shuffle[:int(train_y_shuffle.shape[0]*val_rate)]
    ################
    return train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x

def get_testx(input_date_data_size, std_x, mean_x, remove_columns_list=[], nor_flag=True):
    test_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','test.csv'))
    test_df.drop(columns=['id'], inplace=True)
    for remove_columns_name in remove_columns_list:
        test_df.drop(columns=[remove_columns_name], inplace=True)
    
    ## Testing test 資料集需要注意的事情是，我們會以每四筆輸入輸出一組預測結果。也就是 test 資料共有 528 筆資料，因此我們會預測出 132 筆結果。
    test = test_df.to_numpy()
    test_size, feature_size = test.shape
    # 因為 test 資料已經事先切割好範圍，故需要明確切分每段資料
    test_size = test_size//4

    test_x = np.empty([test_size, feature_size * input_date_data_size], dtype = float)
    for idx in range(test_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, test[idx * input_date_data_size + count]])
        test_x[idx, :] = temp_data

    test_x_old = test_x.copy()
    if nor_flag:
    # test 資料也需要照 training 方式做正規化
        for i in range(len(test_x)):
            for j in range(len(test_x[0])):
                if std_x[j] != 0:
                    test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

    return test_x, test_x_old

# 定義多層linear模型，處理資料後進行訓練跟產出預測
def multi_linear_test():
    # 定義 Regression 的類別，多層
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    input_date_data_size = 4
    learning_rate = 0.001
    epochs = 10000
    train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size)
    
    ### Training
    model = LinearRegression_best(feature_size * input_date_data_size, 1)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 多層要用Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('going dark-->')
    for epoch in range(epochs):
        # forward pass and loss
        y_predicted = model(train_x)
        loss = criterion(y_predicted, train_y)
        # backward pass
        loss.backward()
        # update
        optimizer.step()
        # init optimizer
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

    #predicted = model(train_x).detach().numpy()
    #plt.plot(train_x, train_y, 'ro')
    #plt.plot(train_x, predicted, 'bo')
    #plt.show()
    
    ### val predict
    val_predicted = model(val_x)
    #print('++++++++++++++++++++++++++++++++++', type(val_predicted), val_predicted.shape)
    #print('++++++++++++++++++++++++++++++++++', type(val_predicted.detach().numpy()), val_predicted.detach().numpy().shape)
    plt.plot(val_predicted.detach().numpy()[:,0], label='val_predicted')
    plt.plot(val_y.detach().numpy()[:,0], label='val_y')
    plt.title('multi_linear_test shuffle val')
    plt.legend()
    plt.ylabel('close')
    plt.savefig('multi_linear_test_shuffle')
    plt.close()
    
    val_predicted_seq = model(val_x_seq)
    plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
    plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
    plt.title('multi_linear_test seq val')
    plt.legend()
    plt.ylabel('time steps')
    plt.ylabel('close')
    plt.savefig('multi_linear_test_seq')
    plt.close()
    
    print('val_predicted_seq >>> ', val_predicted_seq.detach().numpy()[:,0])
    print('val_y_seq >>> ', val_y_seq.detach().numpy()[:,0])
    np.save('val_predicted_seq', val_predicted_seq.detach().numpy()[:,0])
    np.save('val_y_seq', val_y_seq.detach().numpy()[:,0])
    
    ### Predicting and saving
    test_x, test_x_old = get_testx(input_date_data_size, std_x, mean_x)
    test_x = torch.from_numpy(test_x.astype(np.float32))
    predicted = model(test_x)

    predicted = [x[0] for x in predicted.tolist()]
    print('-->', type(predicted), len(predicted), predicted)
    
    # 看第四天的收盤價跟預測的吻合程度
    #print(len(test_x), len(test_x[0]))
    #plt.plot(predicted, 'r')
    #plt.plot(test_x_old[:,29], 'b')
    #plt.show()
    
    # list 132
    ids = [x for x in range(len(predicted))]
    output_df = pd.DataFrame({'id': ids, 'result': predicted})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_multi_linear_submission.csv'), index=False)

# 定義dnn模型，處理資料後進行訓練，儲存模型並產出預測
def dnn_test():
    save_path = os.path.join('.', 'hw1_StockPricePrediction')
    
    ## Load Data
    train_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','train.csv'))
    # 處理掉不需要的欄位
    train_df.drop(columns=['datetime'], inplace=True)
    
    ## Preprocess 這邊的做法會以前四天的資料，來預測出第五天的收盤(欄位 'close')結果
    # 設定輸入資料的天數範圍
    input_date_data_size = 4
    # 選擇要預測的欄位順序
    output_predict_index = train_df.columns.get_loc('close')
    # 設定 seed
    #torch.manual_seed(1234)
    np.random.seed(1234)
    #tf.set_random_seed(1234)
    
    train = train_df.to_numpy()
    train_size, feature_size = train.shape
    # 以一段時間的資料當作輸入，故資料數量要扣掉輸入天數範圍
    train_size = train_size - input_date_data_size
    print('train_size, feature_size:', train_size, feature_size)
    
    train_x = np.empty([train_size, feature_size * input_date_data_size], dtype = float)
    train_y = np.empty([train_size, 1], dtype = float)

    for idx in range(train_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, train[idx + count]])
        train_x[idx, :] = temp_data
        train_y[idx, 0] = train[idx + input_date_data_size][output_predict_index]
    print('train_x.shape, train_y.shape:', train_x.shape, train_y.shape)
    #for i in range(train_size):
    #    print(i, train_x[i], train_y[i])
    
    #plt.plot(train_y[:, 0])
    #for i in range(8):
    #    plt.plot(train_y[:, i])
    #plt.show()
    
    # Standardize
    mean_x = np.mean(train_x, axis = 0)
    std_x = np.std(train_x, axis = 0)
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            if std_x[j] != 0:
                train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]
    
    def splitData(X,Y,rate):
        X_train = X[int(X.shape[0]*rate):]
        Y_train = Y[int(Y.shape[0]*rate):]
        X_val = X[:int(X.shape[0]*rate)]
        Y_val = Y[:int(Y.shape[0]*rate)]
        return X_train, Y_train, X_val, Y_val
    
    train_x, train_y, val_x, val_y = splitData(train_x, train_y, 0.1)
    print('train_y go', train_y, 'train_y done')
    
    def summarize_diagnostics(history, model_name, save_path):
        # plot loss
        plt.subplot(211)
        plt.title('MSE Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='val')
        plt.legend()
        # plot accuracy
        plt.subplot(212)
        plt.title('accuracy')
        plt.plot(history.history['acc'], color='blue', label='train accuracy')
        plt.legend()
        # save plot to file
        plt.show()
        #plt.savefig(os.path.join(save_path, model_name))
        plt.close()
    
    # dnn
    def dnn_32():
        model = Sequential()
        model.add(tf.keras.Input(shape=(32), batch_size=None))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(1))
        
        ''' shallow
        model = Sequential()
        model.add(tf.keras.Input(shape=(32), batch_size=None))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        '''
        return model
    
    model = dnn_32()
    
    print(train_x.shape, train_y.shape) #(2215, 32) (2215, 1)
    for i in [0, 1000, -1]:
        print('------->', i, train_x[i], train_y[i])
    
    # train
    model.compile(loss='mse', optimizer='adam', metrics=['mae','acc']) #tf.keras.losses.MeanSquaredError()
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    train_history = model.fit(train_x, train_y, epochs=1000, batch_size=1024, callbacks=[earlystop], shuffle=True, validation_split=0.2, verbose=1) #validation_data=(x_test, y_test)
    summarize_diagnostics(train_history, 'dnn_32', save_path)
    # save
    save_model(model, os.path.join(save_path, 'dnn_32.h5'))

    # test
    obj = model.evaluate(val_x, val_y, verbose=0)
    print('-----> obj:', obj) #[5262.667177774073, 70.27753, 0.0]  # mse mae acc
    #print('-----> acc: {:.6f}% <-----'.format(acc*100.0))

    ###-----###
    # predict
    test_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','test.csv')) 
    test_df.drop(columns=['id'], inplace=True)
    test = test_df.to_numpy()
    test_size, feature_size = test.shape
    # 因為 test 資料已經事先切割好範圍，故需要明確切分每段資料
    test_size = test_size//input_date_data_size

    output_predict_index = test_df.columns.get_loc('close')

    test_x = np.empty([test_size, feature_size * input_date_data_size], dtype = float)
    test_y = np.empty([test_size, 1], dtype = float)
    for idx in range(test_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, test[idx * input_date_data_size + count]])
        test_x[idx, :] = temp_data
        test_y[idx, 0] = test[idx + input_date_data_size][output_predict_index]

    '''
    train_x = np.empty([train_size, feature_size * input_date_data_size], dtype = float)
    train_y = np.empty([train_size, 1], dtype = float)

    for idx in range(train_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, train[idx + count]])
        train_x[idx, :] = temp_data
        train_y[idx, 0] = train[idx + input_date_data_size][output_predict_index]
    print('train_x.shape, train_y.shape:', train_x.shape, train_y.shape)
    '''
    
    # test 資料也需要照 training 方式做正規化
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

    output = model.predict(test_x)
    ids = [x for x in range(output.shape[0])]
    output_df = pd.DataFrame({'id': ids, 'result': output[:,0]})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_dnn_submission.csv'), index=False)
    for i in range(output.shape[0]):
        print(output[i, 0], test_y[i, 0]) # np.array (132, 1)

    plt.plot(output, 'r')
    plt.plot(test_y, 'b')
    plt.show()

# 定義lstm模型，處理資料後進行訓練，儲存模型並產出預測
def lstm_test():
    save_path = os.path.join('.', 'hw1_StockPricePrediction')
    input_date_data_size = 4 #4->1
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    train_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','train.csv'))
    train_df.drop(columns=['datetime'], inplace=True)
    output_predict_index = train_df.columns.get_loc('close')
    
    #(5710, 30, 10) #前30天的10個feather
    train = train_df.to_numpy()
    train_size, feature_size = train.shape
    print('origin train_size, feature_size:', train_size, feature_size)
    
    train_x = np.empty([train_size-input_date_data_size, input_date_data_size, feature_size], dtype=float)
    train_y = np.empty([train_size-input_date_data_size, 1], dtype=float)
    print(train_x.shape, train_y.shape)

    for idx in range(train_size-input_date_data_size):
        #temp_data = np.array([])
        for count in range(input_date_data_size):
            #temp_data = np.hstack([temp_data, train[idx + count]])
            train_x[idx, count, :] = train[idx+count, :]
        train_y[idx, 0] = train[idx + input_date_data_size][output_predict_index]
    print('train_x.shape, train_y.shape:', train_x.shape, train_y.shape)
    #for i in [0, 1000, 2000, -1]:
    #    print(i, 'train_x', train_x[i], 'train_y', train_y[i])

    
    ## Standardize
    mean_x = np.mean(train_x, axis = 0)
    std_x = np.std(train_x, axis = 0)
    print('mean_x.shape, std_x.shape:', mean_x.shape, std_x.shape)
    print('mean_x, std_x go', mean_x, std_x, 'mean_x, std_x done')
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            for k in range(train_x.shape[2]):
                if std_x[j,k] != 0:
                    train_x[i,j,k] = (train_x[i,j,k] - mean_x[j,k]) / std_x[j,k]
                else:
                    print('---std_x[j,k]==0---',j,k)
                    train_x[i,j,k] == 0.0
    print('training Standardize done')
    #for i in [0, 1000, 2000, -1]:
    #    print(i, 'train_x', train_x[i], 'train_y', train_y[i])

    
    def splitData(X,Y,rate):
        X_train = X[int(X.shape[0]*rate):]
        Y_train = Y[int(Y.shape[0]*rate):]
        X_val = X[:int(X.shape[0]*rate)]
        Y_val = Y[:int(Y.shape[0]*rate)]
        return X_train, Y_train, X_val, Y_val
    
    train_x, train_y, val_x, val_y = splitData(train_x, train_y, 0.1)
    #print('train_y go', train_y, 'train_y done')

    
    def buildOneToOneModel(shape): ## X_trian: (5710, 30, 10) #前30天的10個feather
        model = Sequential()
        #model.add(LSTM(8, input_length=shape[1], input_dim=shape[2])) #, return_sequences=True
        #model.add(LSTM(8, input_length=shape[1], input_dim=shape[2]))
        model.add(LSTM(8, activation='relu', input_shape=(4, 8)))
        # output shape: (1, 1)
        model.add(Dense(1)) #model.add(TimeDistributed(Dense(1)))
        model.summary()
        return model
    def summarize_diagnostics(history, model_name, save_path):
        # plot loss
        plt.subplot(211)
        plt.title('MSE Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='val')
        plt.legend()
        # save plot to file
        plt.savefig(os.path.join(save_path, model_name))
        plt.close()

    model_name = 'lstm_test'
    #print(train_x.shape, train_y.shape)

    if os.path.isfile(os.path.join(save_path, model_name+'.h5')):
        print(model_name+' is trained, just load')
        model = load_model(os.path.join(save_path, model_name+'.h5'))
    else:
        model = buildOneToOneModel(train_x.shape)
        model.compile(loss="mse", optimizer="adam")#, metrics=['mse'] #tf.keras.losses.MeanSquaredError()
        callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
        train_history = model.fit(train_x, train_y, epochs=1000, batch_size=1024, validation_data=(val_x, val_y), callbacks=[callback])
        summarize_diagnostics(train_history, model_name, save_path)
        save_model(model, os.path.join(save_path, model_name+'.h5'))
    # test
    mse = model.evaluate(val_x, val_y, verbose=0)
    print('-----> mse: {:.6f} <-----'.format(mse))


    
    # predict # predict
    test_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','test.csv')) 
    test_df.drop(columns=['id'], inplace=True)
    output_predict_index = test_df.columns.get_loc('close')
    
    test = test_df.to_numpy()
    test_size, feature_size = test.shape
    # 因為 test 資料已經事先切割好範圍，故需要明確切分每段資料
    test_size = test_size//input_date_data_size

    test_x = np.empty([test_size, input_date_data_size, feature_size], dtype=float)
    for idx in range(test_size):
        for count in range(input_date_data_size):
            test_x[idx, count, :] = test[idx+count, :]
    print('test_x.shape:', test_x.shape)
    for i in [0, -1]:
        print(i, 'train_x', train_x[i], 'train_y', train_y[i])
    
    # test 資料也需要照 training 方式做正規化
    for i in range(test_x.shape[0]):
        for j in range(test_x.shape[1]):
            for k in range(test_x.shape[2]):
                if std_x[j,k] != 0:
                    test_x[i,j,k] = (test_x[i,j,k] - mean_x[j,k]) / std_x[j,k]
    print('testing Standardize done')
    for i in [0, -1]:
        print(i, 'train_x', train_x[i], 'train_y', train_y[i])

    print()
    print()
    print()
    print()
    print()
    print(train_x.shape, train_y.shape) #(2215, 4, 8) (2215, 1)

    output = model.predict(train_x)
    print(type(output), output.shape) #(132, 1)

    ids = [x for x in range(output.shape[0])]
    output_df = pd.DataFrame({'id': ids, 'result': output[:,0]})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_lstm_submission.csv'), index=False)
    for i in range(output.shape[0]):
        if i%1000 == 0:
            print(i, output[i, 0], train_y[i, 0]) # np.array (132, 1)
    plt.plot(output, 'r')
    plt.plot(train_y, 'b')
    plt.show()
    
    #訓練得很不錯，但是測試應該有問題(時間序列是看哪裡?)
    
    day4close = np.empty([528//4], dtype=float)
    for i in range(528//4):
        day4close[i] = test[i*4, 5]
    
    print('+++++', train_x.shape, test_x.shape)
    print('train_x[0]', train_x[0], train_y[0])
    print('train_x[-1]', train_x[-1], train_y[-1])
    print('test_x[0]', test_x[0], day4close[0])
    print('test_x[-1]', test_x[-1], day4close[-1])
    output = model.predict(test_x)
    print(type(output), output.shape) #(132, 1)

    ids = [x for x in range(output.shape[0])]
    output_df = pd.DataFrame({'id': ids, 'result': output[:,0]})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_lstm_submission.csv'), index=False)
    plt.plot(output, 'r')
    plt.plot(day4close, 'b')
    plt.show()
    
    print('done')


def q1(): #不同Learning Rate
    # 定義多層linear模型，處理資料後進行訓練跟產出預測
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    input_date_data_size = 4
    #learning_rate = 0.001
    epochs = 1000
    train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size)
    
    loss_list_list = []
    learning_rate_list = [1.0, 0.1, 0.01, 0.001]
    for learning_rate in learning_rate_list:
        ### Training
        model = LinearRegression_best(feature_size * input_date_data_size, 1)
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # 多層要用Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('going dark-->')
        loss_list = []
        for epoch in range(epochs):
            # forward pass and loss
            y_predicted = model(train_x)
            loss = criterion(y_predicted, train_y)
            # backward pass
            loss.backward()
            # update
            optimizer.step()
            # init optimizer
            optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')
            loss_list.append(loss.item())
        loss_list_list.append(loss_list)
        #predicted = model(train_x).detach().numpy()
        #plt.plot(train_x, train_y, 'ro')
        #plt.plot(train_x, predicted, 'bo')
        #plt.show()
        
        ### val predict
        val_predicted = model(val_x)
        learning_rate_str = str(learning_rate).split('.')[0] + '_' + str(learning_rate).split('.')[1]
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted), val_predicted.shape)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted.detach().numpy()), val_predicted.detach().numpy().shape)
        plt.plot(val_predicted.detach().numpy()[:,0], label='val_predicted')
        plt.plot(val_y.detach().numpy()[:,0], label='val_y')
        plt.title('q1 {} shuffle val'.format(learning_rate_str))
        plt.legend()
        plt.ylabel('close')
        plt.savefig('q1_{}_shuffle_val'.format(learning_rate_str))
        plt.close()
        
        val_predicted_seq = model(val_x_seq)
        plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
        plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
        plt.title('q1 {} seq val'.format(learning_rate_str))
        plt.legend()
        plt.ylabel('time steps')
        plt.ylabel('close')
        plt.savefig('q1_{}_seq_val'.format(learning_rate_str))
        plt.close()
        
        ### Predicting and saving
        test_x, test_x_old = get_testx(input_date_data_size, std_x, mean_x)
        test_x = torch.from_numpy(test_x.astype(np.float32))
        predicted = model(test_x)

        predicted = [x[0] for x in predicted.tolist()]
        print('-->', type(predicted), len(predicted), predicted)
        
        # 看第四天的收盤價跟預測的吻合程度
        #plt.plot(predicted, 'r')
        #plt.plot(test_x_old[:,29], 'b')
        #plt.show()
        
        # list 132
        ids = [x for x in range(len(predicted))]
        output_df = pd.DataFrame({'id': ids, 'result': predicted})
        output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_q1_{:.3f}_e{}_submission.csv'.format(learning_rate, epochs)), index=False)
    for i in range(len(loss_list_list)):
        plt.plot(loss_list_list[i], label='LR='+str(learning_rate_list[i]))
    plt.title('4 LR MSE loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.savefig('q1_plot_all')
    plt.ylim(0, 100)
    plt.savefig('q1_plot_under_100')
    plt.close()
def q2(): #比較取前2 天和前4 天的資料
    # 定義 Regression 的類別，多層
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    #input_date_data_size = 4
    learning_rate = 0.001
    epochs = 1000
    
    for input_date_data_size in [2]: #, 4
        train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size)
        
        ### Training
        model = LinearRegression_best(feature_size * input_date_data_size, 1)
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # 多層要用Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('going dark-->')
        for epoch in range(epochs):
            # forward pass and loss
            y_predicted = model(train_x)
            loss = criterion(y_predicted, train_y)
            # backward pass
            loss.backward()
            # update
            optimizer.step()
            # init optimizer
            optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

        #predicted = model(train_x).detach().numpy()
        #plt.plot(train_x, train_y, 'ro')
        #plt.plot(train_x, predicted, 'bo')
        #plt.show()
        
        ### val predict
        val_predicted = model(val_x)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted), val_predicted.shape)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted.detach().numpy()), val_predicted.detach().numpy().shape)
        plt.plot(val_predicted.detach().numpy()[:,0], label='val_predicted')
        plt.plot(val_y.detach().numpy()[:,0], label='val_y')
        plt.title('q2_{}days shuffle val'.format(str(input_date_data_size)))
        plt.legend()
        plt.ylabel('close')
        plt.savefig('q2_{}days_shuffle_val'.format(str(input_date_data_size)))
        plt.close()
        
        val_predicted_seq = model(val_x_seq)
        plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
        plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
        plt.title('q2_{}days seq val'.format(str(input_date_data_size)))
        plt.legend()
        plt.ylabel('time steps')
        plt.ylabel('close')
        plt.savefig('q2_{}days_seq_val'.format(str(input_date_data_size)))
        plt.close()
        
        ### Predicting and saving
        test_x, test_x_old = get_testx(input_date_data_size, std_x, mean_x)
        test_x = torch.from_numpy(test_x.astype(np.float32))
        predicted = model(test_x)

        predicted = [x[0] for x in predicted.tolist()]
        print('-->', type(predicted), len(predicted), predicted)
        
        # 看第四天的收盤價跟預測的吻合程度
        #print(len(test_x), len(test_x[0]))
        #plt.plot(predicted, 'r')
        #plt.plot(test_x_old[:,13], 'b')
        #plt.show()
        
        # list 132
        ids = [x for x in range(len(predicted))]
        output_df = pd.DataFrame({'id': ids, 'result': predicted})
        output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_q2_{}days_submission.csv'.format(str(input_date_data_size))), index=False)
def q3(): #比較只取部分特徵和取所有特徵的情況下
    # 定義 Regression 的類別，多層
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    input_date_data_size = 4
    learning_rate = 0.001
    epochs = 1000
    remove_type = [
        'quantity',
        'price',
        'price_spread',
        'only_close'
    ]
    remove_columns_list_list = [
        ['shares', 'amount', 'turnover'],
        ['open', 'high', 'low', 'close'],
        ['change'],
        ['shares', 'amount', 'turnover', 'open', 'high', 'low', 'change']
    ]
    for i in range(len(remove_columns_list_list)):
        train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size, remove_columns_list=remove_columns_list_list[i])
        
        ### Training
        model = LinearRegression_best(feature_size * input_date_data_size, 1)
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # 多層要用Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('going dark-->')
        for epoch in range(epochs):
            # forward pass and loss
            y_predicted = model(train_x)
            loss = criterion(y_predicted, train_y)
            # backward pass
            loss.backward()
            # update
            optimizer.step()
            # init optimizer
            optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

        #predicted = model(train_x).detach().numpy()
        #plt.plot(train_x, train_y, 'ro')
        #plt.plot(train_x, predicted, 'bo')
        #plt.show()
        
        ### val predict
        val_predicted = model(val_x)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted), val_predicted.shape)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted.detach().numpy()), val_predicted.detach().numpy().shape)
        plt.plot(val_predicted.detach().numpy()[:,0], label='val_predicted')
        plt.plot(val_y.detach().numpy()[:,0], label='val_y')
        plt.title('q3 {} shuffle val'.format(remove_type[i]))
        plt.legend()
        plt.ylabel('close')
        plt.savefig('q3_{}_shuffle_val'.format(remove_type[i]))
        plt.close()
        
        val_predicted_seq = model(val_x_seq)
        plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
        plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
        plt.title('q3 {} seq val'.format(remove_type[i]))
        plt.legend()
        plt.ylabel('time steps')
        plt.ylabel('close')
        plt.savefig('q3_{}_seq_val'.format(remove_type[i]))
        plt.close()
        
        ### Predicting and saving
        test_x, test_x_old = get_testx(input_date_data_size, std_x, mean_x, remove_columns_list=remove_columns_list_list[i])
        test_x = torch.from_numpy(test_x.astype(np.float32))
        predicted = model(test_x)

        predicted = [x[0] for x in predicted.tolist()]
        print('-->', type(predicted), len(predicted), predicted)
        
        # 看第四天的收盤價跟預測的吻合程度
        #print(len(test_x), len(test_x[0]))
        #plt.plot(predicted, 'r')
        #plt.plot(test_x_old[:,29], 'b')
        #plt.show()
        
        # list 132
        ids = [x for x in range(len(predicted))]
        output_df = pd.DataFrame({'id': ids, 'result': predicted})
        output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_q3_{}_submission.csv'.format(remove_type[i])), index=False)
def q4(): #比較資料在有無Normalization
    # 定義 Regression 的類別，多層
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    input_date_data_size = 4
    learning_rate = 0.001
    epochs = 1000
    for nor_flag in [True, False]:
        train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size, nor_flag=nor_flag)
        
        ### Training
        model = LinearRegression_best(feature_size * input_date_data_size, 1)
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # 多層要用Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('going dark-->')
        for epoch in range(epochs):
            # forward pass and loss
            y_predicted = model(train_x)
            loss = criterion(y_predicted, train_y)
            # backward pass
            loss.backward()
            # update
            optimizer.step()
            # init optimizer
            optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

        #predicted = model(train_x).detach().numpy()
        #plt.plot(train_x, train_y, 'ro')
        #plt.plot(train_x, predicted, 'bo')
        #plt.show()
        
        ### val predict
        if nor_flag:
            nor_flag_str = 'True'
        else:
            nor_flag_str = 'False'
        val_predicted = model(val_x)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted), val_predicted.shape)
        #print('++++++++++++++++++++++++++++++++++', type(val_predicted.detach().numpy()), val_predicted.detach().numpy().shape)
        plt.plot(val_predicted.detach().numpy()[:,0], label='val_predicted')
        plt.plot(val_y.detach().numpy()[:,0], label='val_y')
        plt.title('q4 {} shuffle val'.format(nor_flag_str))
        plt.legend()
        plt.ylabel('close')
        plt.savefig('q4_{}_shuffle_val'.format(nor_flag_str))
        plt.close()
        
        val_predicted_seq = model(val_x_seq)
        plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
        plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
        plt.title('q4 {} seq val'.format(nor_flag_str))
        plt.legend()
        plt.ylabel('time steps')
        plt.ylabel('close')
        plt.savefig('q4_{}_seq_val'.format(nor_flag_str))
        plt.close()
        
        ### Predicting and saving
        test_x, test_x_old = get_testx(input_date_data_size, std_x, mean_x, nor_flag=nor_flag)
        test_x = torch.from_numpy(test_x.astype(np.float32))
        predicted = model(test_x)

        predicted = [x[0] for x in predicted.tolist()]
        print('-->', type(predicted), len(predicted), predicted)
        
        # 看第四天的收盤價跟預測的吻合程度
        #print(len(test_x), len(test_x[0]))
        #plt.plot(predicted, 'r')
        #plt.plot(test_x_old[:,29], 'b')
        #plt.show()
        
        # list 132
        ids = [x for x in range(len(predicted))]
        output_df = pd.DataFrame({'id': ids, 'result': predicted})
        output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_q4_{}_submission.csv'.format(nor_flag_str)), index=False)
def q5(): #超越Baseline 的Model(在原有測試完超參數的情況下), 加入新特徵(K棒系列特徵)
    # 定義 Regression 的類別，多層
    class LinearRegression_best(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_best, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # 設定輸入資料的天數範圍, lr, epochs
    input_date_data_size = 4
    learning_rate = 0.01
    epochs = 10000
    ########################################################################
    ## Load Data, 處理掉不需要的欄位
    train_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','train.csv'))
    train_df.drop(columns=['datetime'], inplace=True)
    
    ## Preprocess 以前四天的資料預測第五天的收盤(欄位 'close')結果
    # 選擇要預測的欄位順序
    output_predict_index = train_df.columns.get_loc('close')
    
    #print(train_df.head(50), train_df.shape) # shares amount open high  low close change turnover #(2465, 8)
    #print(train_df['change'], type(train_df['change']), train_df['change'].mean(), train_df['change'].std(), type(train_df['change'].std()))

    # make MA5
    MA5_list = []
    # make K line
    red_upper_line_list = []
    red_under_line_list = []
    green_upper_line_list = []
    green_under_line_list = []
    cross_upper_line_list = []
    cross_under_line_list = []
    change_std = train_df['change'].std()
    print('change_std:', change_std)
    
    for i in range(train_df.shape[0]):
        # make MA5
        if i < 4:
            close_avg = 0.0
            for j in range(0, i+1):
                close_avg += train_df['close'][j]
            close_avg /= (i+1)
        else:
            close_avg = 0.0
            for j in range(i-4, i+1):
                close_avg += train_df['close'][j]
            close_avg /= 5.0
        MA5_list.append(close_avg)
        
        # make K line
        if -change_std < train_df['change'][i] < change_std: #十字線
            #print(train_df['change'][i], '十字線')
            red_upper_line_list.append(0.0)
            red_under_line_list.append(0.0)
            green_upper_line_list.append(0.0)
            green_under_line_list.append(0.0)
            cross_upper_line_list.append( train_df['high'][i]-(train_df['open'][i]+train_df['close'][i])/2 )
            cross_under_line_list.append( (train_df['open'][i]+train_df['close'][i])/2 - train_df['low'][i] )
        elif -change_std >= train_df['change'][i]: #跌
            #print(train_df['change'][i], '跌')
            red_upper_line_list.append(0.0)
            red_under_line_list.append(0.0)
            green_upper_line_list.append( train_df['high'][i]-train_df['open'][i] )
            green_under_line_list.append( train_df['close'][i]-train_df['low'][i] )
            cross_upper_line_list.append(0.0)
            cross_under_line_list.append(0.0)
        elif change_std <= train_df['change'][i]: #漲
            #print(train_df['change'][i], '漲')
            red_upper_line_list.append( train_df['high'][i]-train_df['close'][i] )
            red_under_line_list.append( train_df['open'][i]-train_df['low'][i] )
            green_upper_line_list.append(0.0)
            green_under_line_list.append(0.0)
            cross_upper_line_list.append(0.0)
            cross_under_line_list.append(0.0)
    '''
    print('red_upper_line_list', red_upper_line_list, len(red_upper_line_list))
    print('red_under_line_list', red_under_line_list, len(red_under_line_list))
    print('green_upper_line_list', green_upper_line_list, len(green_upper_line_list))
    print('green_under_line_list', green_under_line_list, len(green_under_line_list))
    print('cross_upper_line_list', cross_upper_line_list, len(cross_upper_line_list))
    print('cross_under_line_list', cross_under_line_list, len(cross_under_line_list))
    '''
    train_df['MA5'] = MA5_list
    train_df['red_upper_line'] = red_upper_line_list
    train_df['red_under_line'] = red_under_line_list
    train_df['green_upper_line'] = green_upper_line_list
    train_df['green_under_line'] = green_under_line_list
    train_df['cross_upper_line'] = cross_upper_line_list
    train_df['cross_under_line'] = cross_under_line_list
    #print(train_df.head(1000), train_df.shape)

    #for remove_columns_name in remove_columns_list:
    #    train_df.drop(columns=[remove_columns_name], inplace=True)
    
    # 設定 seed
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    train = train_df.to_numpy()
    train_size, feature_size = train.shape
    # 以一段時間的資料當作輸入，故資料數量要扣掉輸入天數範圍
    train_size = train_size - input_date_data_size
    print('train_size, feature_size:', train_size, feature_size)
    
    train_x = np.empty([train_size, feature_size * input_date_data_size], dtype = float)
    train_y = np.empty([train_size, 1], dtype = float)

    for idx in range(train_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, train[idx + count]])
        train_x[idx, :] = temp_data
        train_y[idx, 0] = train[idx + input_date_data_size][output_predict_index]
    print('train_x.shape, train_y.shape:', train_x.shape, train_y.shape)
    
    ## Standardize
    mean_x = np.mean(train_x, axis = 0)
    std_x = np.std(train_x, axis = 0)
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            if std_x[j] != 0:
                train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]

    ## Training
    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    train_y = train_y.view(train_y.shape[0], 1)
    
    val_rate = 0.1
    val_x_seq, val_y_seq = train_x[int(train_x.shape[0]*(1-val_rate)):], train_y[int(train_y.shape[0]*(1-val_rate)):]
    train_x, train_y = train_x[:int(train_x.shape[0]*(1-val_rate))], train_y[:int(train_y.shape[0]*(1-val_rate))]
    
    ########################################################################
    
    #train_x, train_y, val_x, val_y, val_x_seq, val_y_seq, feature_size, std_x, mean_x = get_trainx_trainy(input_date_data_size)
    
    ### Training
    model = LinearRegression_best(feature_size * input_date_data_size, 1)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 多層要用Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('going dark-->')
    loss_list = []
    for epoch in range(epochs):
        # forward pass and loss
        y_predicted = model(train_x)
        loss = criterion(y_predicted, train_y)
        # backward pass
        loss.backward()
        # update
        optimizer.step()
        # init optimizer
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')
        loss_list.append(loss.item())
    
    plt.plot(loss_list, label='MSE loss')
    plt.title('4 LR MSE loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.savefig('q5_plot_all')
    plt.ylim(0, 100)
    plt.savefig('q5_plot_under_100')
    plt.ylim(0, 10)
    plt.savefig('q5_plot_under_10')
    plt.close()
    #predicted = model(train_x).detach().numpy()
    #plt.plot(train_x, train_y, 'ro')
    #plt.plot(train_x, predicted, 'bo')
    #plt.show()
    
    ### val predict
    val_predicted_seq = model(val_x_seq)
    plt.plot(val_predicted_seq.detach().numpy()[:,0], label='val_predicted_seq')
    plt.plot(val_y_seq.detach().numpy()[:,0], label='val_y_seq')
    plt.title('q5 seq val')
    plt.legend()
    plt.ylabel('time steps')
    plt.ylabel('close')
    plt.savefig('q5_seq_val')
    plt.close()
    
    ### Predicting and saving
    
    test_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','test.csv'))
    test_df.drop(columns=['id'], inplace=True)
    
    # make MA5
    MA5_list = []
    # make K line
    red_upper_line_list = []
    red_under_line_list = []
    green_upper_line_list = []
    green_under_line_list = []
    cross_upper_line_list = []
    cross_under_line_list = []
    #change_std = test_df['change'].std() #8.19109309305959
    print('change_std:', change_std)
    
    for i in range(test_df.shape[0]):
        # make MA5
        close_avg = 0.0
        counter = 0
        for j in range(0, i%5+1):
            if i+j >= test_df.shape[0]:
                continue
            else:
                close_avg += test_df['close'][i+j]
                counter += 1
        close_avg /= counter
        MA5_list.append(close_avg)
        
        # make K line
        if -change_std < test_df['change'][i] < change_std: #十字線
            print(test_df['change'][i], '十字線')
            red_upper_line_list.append(0.0)
            red_under_line_list.append(0.0)
            green_upper_line_list.append(0.0)
            green_under_line_list.append(0.0)
            cross_upper_line_list.append( test_df['high'][i]-(test_df['open'][i]+test_df['close'][i])/2 )
            cross_under_line_list.append( (test_df['open'][i]+test_df['close'][i])/2 - test_df['low'][i] )
        elif -change_std >= test_df['change'][i]: #跌
            print(test_df['change'][i], '跌')
            red_upper_line_list.append(0.0)
            red_under_line_list.append(0.0)
            green_upper_line_list.append( test_df['high'][i]-test_df['open'][i] )
            green_under_line_list.append( test_df['close'][i]-test_df['low'][i] )
            cross_upper_line_list.append(0.0)
            cross_under_line_list.append(0.0)
        elif change_std <= test_df['change'][i]: #漲
            print(test_df['change'][i], '漲')
            red_upper_line_list.append( test_df['high'][i]-test_df['close'][i] )
            red_under_line_list.append( test_df['open'][i]-test_df['low'][i] )
            green_upper_line_list.append(0.0)
            green_under_line_list.append(0.0)
            cross_upper_line_list.append(0.0)
            cross_under_line_list.append(0.0)
    
    print('red_upper_line_list', red_upper_line_list, len(red_upper_line_list))
    print('red_under_line_list', red_under_line_list, len(red_under_line_list))
    print('green_upper_line_list', green_upper_line_list, len(green_upper_line_list))
    print('green_under_line_list', green_under_line_list, len(green_under_line_list))
    print('cross_upper_line_list', cross_upper_line_list, len(cross_upper_line_list))
    print('cross_under_line_list', cross_under_line_list, len(cross_under_line_list))
    
    test_df['MA5'] = MA5_list
    test_df['red_upper_line'] = red_upper_line_list
    test_df['red_under_line'] = red_under_line_list
    test_df['green_upper_line'] = green_upper_line_list
    test_df['green_under_line'] = green_under_line_list
    test_df['cross_upper_line'] = cross_upper_line_list
    test_df['cross_under_line'] = cross_under_line_list
    print(test_df.head(1000), test_df.shape)

    
    ## Testing test 資料集需要注意的事情是，我們會以每四筆輸入輸出一組預測結果。也就是 test 資料共有 528 筆資料，因此我們會預測出 132 筆結果。
    test = test_df.to_numpy()
    test_size, feature_size = test.shape
    # 因為 test 資料已經事先切割好範圍，故需要明確切分每段資料
    test_size = test_size//input_date_data_size

    test_x = np.empty([test_size, feature_size * input_date_data_size], dtype = float)
    for idx in range(test_size):
        temp_data = np.array([])
        for count in range(input_date_data_size):
            temp_data = np.hstack([temp_data, test[idx * input_date_data_size + count]])
        test_x[idx, :] = temp_data

    # test 資料也需要照 training 方式做正規化
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    
    test_x = torch.from_numpy(test_x.astype(np.float32))
    predicted = model(test_x)

    predicted = [x[0] for x in predicted.tolist()]
    print('-->', type(predicted), len(predicted), predicted)
    
    # list 132
    ids = [x for x in range(len(predicted))]
    output_df = pd.DataFrame({'id': ids, 'result': predicted})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_q5_submission.csv'), index=False)
    
    
if __name__ == "__main__":
    # 0, model ability test
    #multi_linear_test()
    #dnn_test()
    #lstm_test()
    
    ## 對multi_linear()做各種參數變化並觀察分數
    ### 改middle
    ### 10.24172{learning_rate=0.01, epochs=100, middle=600}
    ### 10.45073{learning_rate=0.01, epochs=100, middle=6000} middle 60 or 6 都很差
    ### 11.22146{learning_rate=0.01, epochs=100, middle=1000}
    ### 10.68334{learning_rate=0.01, epochs=100, middle=200}
    
    ### 改epochs
    ### 9.79706{learning_rate=0.01, epochs=1000, middle=600}
    ### 9.41729{learning_rate=0.01, epochs=10000, middle=600}
    ### 18.76628{learning_rate=0.01, epochs=1000, middle=2000}
    
    ### 改lr
    ### 9.88990{learning_rate=0.1, epochs=1000, middle=600}
    ### 9.40615{learning_rate=0.001, epochs=10000, middle=600}
    
    ### 改三層 
    ### 9.74213{learning_rate=0.01, epochs=1000, middle=600,600}
    ### 9.34241{learning_rate=0.001, epochs=10000, middle=600,600}
    ### 9.32999{learning_rate=0.001, epochs=10000, middle=600,1200} -- best!!
    ### 9.34474{learning_rate=0.001, epochs=10000, middle=1200,600}
    ### 16.16026{learning_rate=0.001, epochs=10000, middle=1200,1200}
    ### 加reLU效果很差
    
    
    
    # 1, diff Learning Rate
    #q1()
    ### 9.79152{learning_rate=0.1, epochs=1000, middle=600,1200}
    ### 9.76572{learning_rate=0.01, epochs=1000, middle=600,1200}
    ### 10.11540{learning_rate=0.001, epochs=1000, middle=600,1200}
    ### 116.05683{learning_rate=0.1, epochs=100, middle=600,1200}
    ### 11.83569{learning_rate=0.01, epochs=100, middle=600,1200}
    ### 10.06587{learning_rate=0.001, epochs=100, middle=600,1200}
    
    # 2, 2days vs. 4days
    #q2()
    ### 143.49388{learning_rate=0.001, epochs=1000, middle=600,1200, 2dayData}
    ### 10.11540{learning_rate=0.001, epochs=1000, middle=600,1200, 4dayData}
    
    # 3, all features vs. part features
    #q3()
    ### 9.82741{learning_rate=0.001, epochs=1000, middle=600,1200, no quantity Data(shares amount turnover)}
    ### 163.29943{learning_rate=0.001, epochs=1000, middle=600,1200, no price Data(open high low close)}
    ### 9.48439{learning_rate=0.001, epochs=1000, middle=600,1200, no price spread Data(change)}
    ### 9.90882{learning_rate=0.001, epochs=1000, middle=600,1200, only close Data}
    
    # 4, Normalization or not
    #q4()
    ### 10.11540{learning_rate=0.001, epochs=1000, middle=600,1200, Normalization}
    ### 1006381.86377{learning_rate=0.001, epochs=1000, middle=600,1200, no Normalization}
    
    # 5, best Model implement 
    # 加入新特徵(K棒系列特徵)
    q5()
    ### 12.07070{learning_rate=0.01, epochs=10000, middle=600,1200, add K棒系列特徵 and 5MA}