# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:00:06 2024

@author: 赵书婷
"""

import matplotlib
import warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.layers import MaxPooling1D, Conv1D,Flatten
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import os
import glob
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import time
from sklearn.model_selection import GridSearchCV

def CNN_CH4(filename,comp):
    filename_base = filename.split('\\')[1]
    df = pd.read_csv(filename)
    for c in comp:
        print('------------------------------------------------------------')
        new_col_name = '_'.join(comp[c][:-1])
        print("Input:",comp[c])
        data = df[comp[c]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        sacler_data_len = len(data.columns)
        def splitData(var, per_test):
            num_test = int(len(var) * per_test)
            train_size = int(len(var) - num_test)
            train_data = var[0:train_size]
            test_data = var[train_size:train_size + num_test]
            return train_data, test_data
        
        df_training, df_testing = splitData(scaled_data, 0.2)
        print(df_training.shape)
        print(df_testing.shape)
    
    
        def createXY(data, n_past, n_steps_out):
            dataX, dataY = list(), list()
            for i in range(len(data)):
                end_ix = i + n_past  ## 0+3=3,1+3=4,...,19618+3=19621
                out_end_ix = end_ix + n_steps_out  ## 3+3=6,4+3=7,...,19621+3=19624
                if out_end_ix > len(data):  ## 6 < len(data),7<len(data),...,19623+3=19626>len(data)=19624
                    # print("------------out_end_ix of end",out_end_ix,'---------------')
                    break
                dataX.append(data[i:end_ix,0:data.shape[1]]) ## 0:3,0:7;1:4,0:7
                dataY.append(data[end_ix:out_end_ix,sacler_data_len-1]) ##3:6,6;4:7,6
            return array(dataX), array(dataY)
        
        n_past = 1
        n_output = 1
        trainX, trainY = createXY(df_training, n_past, n_output)
        testX, testY = createXY(df_testing, n_past, n_output)
        
        print('train Shape---', trainX.shape)
        print('trainY Shape---', trainY.shape)
        print('testX Shape---', testX.shape)
        print('testY Shape---', testY.shape)
    
        def CnnLSTM(optimizer='adam', batch_size=32, epochs=40):
            model = Sequential()
            model.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(MaxPooling1D(pool_size=1))
            model.add(Flatten())
            model.add(Dropout(0.1))
            model.add(Dense(trainY.shape[1]))
            model.compile(optimizer=optimizer, loss='mse', metrics='accuracy')
            return model
        CNN_LSTM_Model = KerasRegressor(CnnLSTM, epochs=40, verbose=1)        
        param_grid = {
            'optimizer': ['adam'],
            'batch_size': [32, 64, 128],
            'epochs': [20, 30, 40]
        }
        grid = GridSearchCV(estimator=CNN_LSTM_Model, param_grid=param_grid, cv=2)
        grid_result = grid.fit(trainX, trainY)
        best_params = grid_result.best_params_
        
        best_params
        print('best_params:', best_params)
        
        best_model = grid_result.best_estimator_
        print('Result_Ana_Function:', testX.shape)
        y_pred_train = best_model.predict(trainX)
        y_pred_train = np.array(y_pred_train).reshape(-1, 1)
        prediction_copies_array_train = np.repeat(y_pred_train, sacler_data_len, axis=-1)
        y_pred_train = scaler.inverse_transform(
            np.reshape(prediction_copies_array_train, (len(prediction_copies_array_train), sacler_data_len))) 
        pred_train = y_pred_train[:, sacler_data_len - 1]
        train_y_data = np.array(trainY).reshape(-1, 1)
        original_copies_array_train = np.repeat(train_y_data, sacler_data_len, axis=-1)
        y_true_train = scaler.inverse_transform(
            np.reshape(original_copies_array_train, (len(original_copies_array_train), sacler_data_len)))[:,
                 sacler_data_len - 1]
        y_trues_train = y_true_train
        preds_train = pred_train
        plt.plot(y_trues_train, color='red', label='Real Value')
        plt.plot(preds_train, color='blue', label='Pred Value')
        plt.title('Prediction et0')
        plt.xlabel('Time (h)')
        plt.ylabel('ET0')
        plt.legend()
        plt.show()
        print("--------","训练集结果","---------------")
        print('MSE:', mse(y_trues_train, preds_train))
        print('MAE:', mae(y_trues_train, preds_train))
        print('R²:', r2(y_trues_train, preds_train))
        print('RMSE:', np.sqrt(mse(y_trues_train, preds_train)))
        y_pred = best_model.predict(testX)
        y_pred = np.array(y_pred).reshape(-1, 1)
        prediction_copies_array = np.repeat(y_pred, sacler_data_len, axis=-1)
        pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction_copies_array), sacler_data_len)))  
        pred = pred[:, sacler_data_len - 1]
        test_data = np.array(testY).reshape(-1, 1)
        original_copies_array = np.repeat(test_data, sacler_data_len, axis=-1)
        y_true = scaler.inverse_transform(np.reshape(original_copies_array, (len(original_copies_array), sacler_data_len)))[:,
                 sacler_data_len - 1]
        
        y_trues = y_true
        preds = pred
        plt.plot(y_trues, color='red', label='Real Value')
        plt.plot(preds, color='blue', label='Pred Value')
        plt.title('Prediction et0')
        plt.xlabel('Time (h)')
        plt.ylabel('ET0')
        plt.legend()
        plt.show()
        print("--------", "测试集结果", "---------------")
        print('MSE:', mse(y_trues, preds))
        print('MAE:', mae(y_trues, preds))
        print('R²:', r2(y_trues, preds))
        print('RMSE:', np.sqrt(mse(y_trues, preds)))
        merge_result = np.concatenate((y_trues, preds),axis=0)
        result_save_file_pred = pd.DataFrame(y_trues, preds)
        result_save_file_pred.to_csv('D:/seven/results/'+str(i)+'d/cnn1/'+filename_base)
        print('------------------------------------------------------------')
if __name__ == '__main__':
    start_time = time.time()
    comp = {
        'C1_1A': ['tmax_y','tmin_y','dswrf','ws_jz','p_jz','vpd_jz', 'FCH4_F_ANNOPTLM']
    }
    for i in range(1,4):
        file_path = "D:/seven/new/"+str(i)+"d/"
        acquire_file = glob.glob(os.path.join(file_path, '*.csv'))
    
        for filename in acquire_file:
            CNN_CH4(filename,comp)
        end_time = time.time()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(f"代码运行时间: {formatted_time} 时: 分: 秒")