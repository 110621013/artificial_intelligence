
#對股市K棒多理解並做出新feather
'''
不同模型比較(linear dnn LSTM) -> 完成, linear勝出, dnn跟LSTM以及有reLU的linear都會有強烈高估
不同Learning Rate
 比較取前2 天和前4 天的資料
 比較只取部分特徵和取所有特徵的情況下
 比較資料在有無Normalization
請說明你超越Baseline 的Model是如何實作的。
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

def multi_linear():
    # 定義 Regression 的類別，多層
    class LinearRegression_v2(nn.Module):
        def __init__(self, input_dim, output_dim): #32,1
            super(LinearRegression_v2, self).__init__()
            # 定義每層用什麼樣的形式
            self.layer1 = torch.nn.Linear(input_dim, 600)
            self.layer2 = torch.nn.Linear(600, 1200)
            self.layer3 = torch.nn.Linear(1200, output_dim)
        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            return x
    
    ## Load Data
    train_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','train.csv'))
    test_df = pd.read_csv(os.path.join('.','hw1_StockPricePrediction','Dataset','test.csv')) 
    # 處理掉不需要的欄位
    train_df.drop(columns=['datetime'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)
    
    ## Preprocess 這邊的做法會以前四天的資料，來預測出第五天的收盤(欄位 'close')結果
    # 設定輸入資料的天數範圍
    input_date_data_size = 4
    # 選擇要預測的欄位順序
    output_predict_index = train_df.columns.get_loc('close')
    # 設定 seed
    torch.manual_seed(1234)
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
    
    model = LinearRegression_v2(feature_size * input_date_data_size, 1)
    learning_rate = 0.001
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    print('going dark-->')
    epochs = 1000
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

    predicted = model(train_x).detach().numpy()
    plt.plot(train_x, train_y, 'ro')
    plt.plot(train_x, predicted, 'bo')
    plt.show()
    
    
    
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

    test_x_old = test_x.copy()
    # test 資料也需要照 training 方式做正規化
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

    test_x = torch.from_numpy(test_x.astype(np.float32))
    predicted = model(test_x)

    predicted = [x[0] for x in predicted.tolist()]
    print('-->', type(predicted), len(predicted), predicted)
    
    print(len(test_x), len(test_x[0]))
    plt.plot(predicted, 'r')
    plt.plot(test_x_old[:,29], 'b')
    plt.show()
    
    # list 132
    ids = [x for x in range(len(predicted))]
    output_df = pd.DataFrame({'id': ids, 'result': predicted})
    output_df.to_csv(os.path.join('.','hw1_StockPricePrediction','hw1_multi_linear_submission.csv'), index=False)

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
    
    
    
if __name__ == "__main__":
    multi_linear()
    # 改middle
    # 10.24172{learning_rate=0.01, epochs=100, middle=600}
    # 10.45073{learning_rate=0.01, epochs=100, middle=6000} middle 60 or 6 都很差
    # 11.22146{learning_rate=0.01, epochs=100, middle=1000}
    # 10.68334{learning_rate=0.01, epochs=100, middle=200}
    
    # 改epochs
    # 9.79706{learning_rate=0.01, epochs=1000, middle=600}
    # 9.41729{learning_rate=0.01, epochs=10000, middle=600}
    # 18.76628{learning_rate=0.01, epochs=1000, middle=2000}
    
    # 改lr
    # 9.88990{learning_rate=0.1, epochs=1000, middle=600}
    # 9.40615{learning_rate=0.001, epochs=10000, middle=600}
    
    # 改三層 
    # 9.74213{learning_rate=0.01, epochs=1000, middle=600,600}
    # 9.34241{learning_rate=0.001, epochs=10000, middle=600,600}
    # 9.32999{learning_rate=0.001, epochs=10000, middle=600,1200}
    # 9.34474{learning_rate=0.001, epochs=10000, middle=1200,600}
    # 16.16026{learning_rate=0.001, epochs=10000, middle=1200,1200}
    # 加reLU效果很差
    
    #dnn_test()
    #lstm_test()