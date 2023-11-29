
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from keras.metrics import binary_crossentropy

from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster

import statsmodels.api as sm 
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
import tensorflow as tf


    # In[3]:


    #df = pd.read_csv("C:\\Users\\carso\\DP-LSTM-Differential-Privacy-inspired-LSTM-for-Stock-Prediction-Using-Financial-News-1\\data\\source_price.csv")
def dp_lstm(data):
    df_train = data.train_set[0]
    df_test = data.dev_set[0]
    df_val = data.test_set[0]

    var_train = np.var(df_train)
    var_test = np.var(df_test)
    var_val = np.var(df_val)

    mu = 0
    noise = 0.1

    sigma_train = noise * var_train
    sigma_test = noise * var_test
    sigma_val = noise * var_val

    df_train_noise = df_train.copy()
    df_test_noise = df_test.copy()
    df_val_noise = df_val.copy()

    for idx in range(len(df_train_noise)):
        df_train_noise[idx] += np.random.normal(mu, sigma_train)
    for idx in range(len(df_test_noise)):
        df_train_noise[idx] += np.random.normal(mu, sigma_test)
    for idx in range(len(df_val_noise)):
        df_val_noise[idx] += np.random.normal(mu, sigma_val)

    train_sequence_length = 50
    test_sequence_length = 50
    val_sequence_length = 50
    normalise= True
    batch_size=100
    neurons=50
    epochs=5
    prediction_len=1
    dense_output=1
    drop_out=0
    val_mse_list = []
    test_mse_list = []

    data_windows = []
    len_test = len(df_test_noise)
    for i in range(len_test - test_sequence_length):
        data_windows.append(df_test_noise[i:i+test_sequence_length])

    data_windows = np.array(data_windows).astype(float)
    y_test_ori = data_windows[:, -1, [0]]

    val_data_windows = []
    len_val = len(df_val_noise)
    for i in range(len_val - val_sequence_length):
        val_data_windows.append(df_val_noise[i:i+val_sequence_length])

    val_data_windows = np.array(val_data_windows).astype(float)
    y_val_ori = val_data_windows[:, -1, [0]]

    train_data_windows = []
    len_train = len(df_train_noise)
    for i in range(len_train - train_sequence_length):
        train_data_windows.append(df_train_noise[i:i+train_sequence_length])
    
    model_train_df = np.array(train_data_windows).astype(float)

    # for stock in range(len(model_train_df)):
    model_label = model_train_df[:, -1, [0]]
    window_data=data_windows
    win_num=window_data.shape[0]
    col_num=window_data.shape[2]
    normalised_data = []
    record_min=[]
    record_max=[]
    for win_i in range(0,win_num):
        normalised_window = []
        for col_i in range(0,col_num):#col_num):
            temp_col=window_data[win_i,:,col_i]
            temp_min=min(temp_col)
            if col_i==0:
                record_min.append(temp_min)#record min
            temp_col=temp_col-temp_min
            temp_max=max(temp_col)
            if col_i==0:
                record_max.append(temp_max)#record max
            temp_col=temp_col/temp_max
            normalised_window.append(temp_col)
        for col_i in range(1,col_num):
            temp_col=window_data[win_i,:,col_i]
            normalised_window.append(temp_col)
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    normalised_data=np.array(normalised_data)

    val_window_data=val_data_windows
    val_win_num=val_window_data.shape[0]
    val_col_num=val_window_data.shape[2]
    val_normalised_data = []
    val_record_min=[]
    val_record_max=[]

    for win_i in range(0,val_win_num):
        normalised_window = []
        for col_i in range(0,col_num):#col_num):
            temp_col=val_window_data[win_i,:,col_i]
            temp_min=min(temp_col)
            if col_i==0:
                val_record_min.append(temp_min)#record min
            temp_col=temp_col-temp_min
            temp_max=max(temp_col)
            if col_i==0:
                val_record_max.append(temp_max)#record max
            temp_col=temp_col/temp_max
            normalised_window.append(temp_col)
        for col_i in range(1,val_col_num):
            temp_col=val_window_data[win_i,:,col_i]
            normalised_window.append(temp_col)
        normalised_window = np.array(normalised_window).T
        val_normalised_data.append(normalised_window)
    val_normalised_data=np.array(val_normalised_data)



    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(model_train_df.shape[1], model_train_df.shape[2]), return_sequences = True))
    model.add(Dropout(drop_out))
    model.add(LSTM(neurons,return_sequences = True))
    model.add(LSTM(neurons,return_sequences = False))
    model.add(Dropout(drop_out))
    model.add(Dense(dense_output, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error',
                    optimizer='adam')
    
    # Fit the model
    model.fit(model_train_df, model_label, epochs=epochs,batch_size=batch_size)

    # In[16]:


    #multi sequence predict
    model_data = data_windows
    prediction_seqs = []
    window_size=test_sequence_length
    pre_win_num=int(len(model_data)/prediction_len)
    for i in range(0,pre_win_num):
        curr_frame = model_data[i*prediction_len]
        predicted = []
        for _ in range(0,prediction_len):
            temp = model.predict(curr_frame[newaxis,:,:])[0]
            predicted.append(temp)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    #de_predicted
    de_predicted=[]
    len_pre_win=int(len(model_data)/prediction_len)
    len_pre=prediction_len

    m=0
    for i in range(0,len_pre_win):
        for j in range(0,len_pre):
            de_predicted.append(prediction_seqs[i][j][0]*record_max[m]+record_min[m])
            m=m+1

    error = []
    diff = np.asarray(data.dev_label[0]).shape[0]-prediction_len*pre_win_num
    
    for i in range(y_test_ori.shape[0]-diff):
        error.append(y_test_ori[i,] - de_predicted[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val) 
        absError.append(abs(val))

    error_percent=[]
    for i in range(len(error)):
        val= error[i] / ((y_test_ori[i,] + de_predicted[i]) / 2)
        new_val=abs(val)
        error_percent.append(new_val)

    test_MSE =sum(squaredError) / len(squaredError)
    print('test_MSE: ', test_MSE)
    #test_mse_list.append(test_MSE)

    model_data = val_data_windows
    prediction_seqs = []
    window_size = val_sequence_length
    pre_win_num = int(len(model_data)/prediction_len)

    for i in range(0,pre_win_num):
        curr_frame = model_data[i*prediction_len]
        predicted = []
        for j in range(0,prediction_len):
            temp = model.predict(curr_frame[newaxis,:,:])[0]
            predicted.append(temp)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    #de_predicted
    de_predicted=[]
    len_pre_win=int(len(model_data)/prediction_len)
    len_pre=prediction_len
    m=0
    for i in range(0,len_pre_win):
        for j in range(0,len_pre):
            de_predicted.append(prediction_seqs[i][j][0]*val_record_max[m]+val_record_min[m])
            m=m+1
    error = []
    diff = np.asarray(data.test_label).shape[0]-prediction_len*pre_win_num

    for i in range(y_val_ori.shape[0]-diff):
        error.append(y_val_ori[i,] - de_predicted[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val) 
        absError.append(abs(val))

    error_percent=[]
    for i in range(len(error)):
        val= (absError[i] + y_val_ori[i,]) / 2
        val=abs(val)
        error_percent.append(val)

    val_MSE=sum(squaredError) / len(squaredError)

    val_mse_list.append(val_MSE)
    print('val_MSE: ', val_MSE)

    #return(f'Test MSE: {sum(test_mse_list) / len(test_mse_list)}, Val MSE: {sum(val_mse_list) / len(val_mse_list)}')

