from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import math
import numpy as np
import random
import tensorflow as tf
from os import path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from pandas import read_csv, DataFrame
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Concatenate, Lambda, TimeDistributed
from keras import optimizers, initializers
from numpy import concatenate
from keras import backend as K

def preprocess(name, window_size, predict_size):
    data_frame = pd.read_csv(name)
    del data_frame['time']
    data_values = data_frame.values
    num_series = data_frame.shape[1]
    time_len = data_frame.shape[0]
    #duplicate the last time step to avoid out of bound error
    data_values = np.insert(data_values, time_len, data_values[time_len-1],axis=0)
    covariant = np.zeros((time_len, 5))
    #add dist to 1st obs, day, hour, week, month in order
    for i in range(time_len):
        covariant[i][0] = i/time_len
        covariant[i][1] = i%7/7
        covariant[i][2] = i%24/24
        covariant[i][3] = int(i%365/7)/52
        covariant[i][4] = int(i%365/30.5)/12
    #processed shape = (batch_size, window_size, #variables)
    batch_size = (int((time_len-window_size)/predict_size)+1)*num_series
    print("time_len: {}".format(time_len))
    print("batch_size: {}".format(batch_size))
    print("for loop size: {}".format(int((batch_size/num_series-1)*predict_size+window_size)))
    print(data_values.shape)
    print(data_values[0:0+window_size,0].shape)
    processed_data = np.zeros((batch_size, window_size, 9))

    #processed: z(t+1), scaled-z(t+1), scaled-z(t), covariants, #series
    for i in range(int((time_len-window_size)/predict_size)+1):
        for j in range(num_series):
            processed_data[i*num_series+j] = np.column_stack([data_values[i*predict_size+1:i*predict_size+window_size+1,j].reshape(window_size, 1),
            data_values[i*predict_size+1:i*predict_size+window_size+1,j].reshape(window_size, 1),
            data_values[i*predict_size:i*predict_size+window_size,j].reshape(window_size, 1),
            covariant[i*predict_size:i*predict_size+window_size, :], np.full((window_size, 1), j)])
    scaleV = np.zeros((batch_size, 2))
    for i in range(batch_size):
        for j in range(window_size):
            scaleV[i,0] += processed_data[i, j, 2]
        scaleV[i,0] /= window_size
        scaleV[i,0] += 1
        for j in range(window_size):
            processed_data[i, j, 1] /= scaleV[i,0]
            processed_data[i, j, 2] /= scaleV[i,0]
    scaleV[:,1] = scaleV[:,0]/scaleV[:,0].sum()
    np.save(name[:name.find('.')]+"_data", processed_data)
    np.save(name[:name.find('.')]+"_scale", scaleV)
    print("processed data:")
    print(processed_data[-10,-1000:,:])
    print("scale V:")
    print(scaleV[-10:])
    return time_len, num_series

def gaussian(z, mu, sigma):
    #a=tf.divide(K.exp(tf.divide(-K.square(z-mu),(K.constant(2)*K.square(sigma)))),K.sqrt(K.constant(2*math.pi)*K.square(sigma)))
    a = tf.contrib.distributions.NormalWithSoftplusScale(mu, sigma)
    prob = a.log_prob(z)
    return prob

def neg_binomial(z, mu, alpha):
    return math.gamma(z+1/alpha)/(math.gamma(z+1)*math.gamma(1/alpha))*pow((1/(1+alpha*mu)),1/alpha)*pow(alpha*mu/(1+alpha*mu),z)

def log_likelihood(y_true, para_pred):
    likelihood = gaussian(y_true[:,:,1], para_pred[:, :, 0], para_pred[:, :, 1])
    a = K.mean(likelihood, axis = 1)
    return -a

def ND_metrics(y_true, y_pred):
    return np.sum(np.absolute(y_pred-y_true))/np.sum(np.absolute(y_true))

def RMSE_metrics(y_true, y_pred, predict_size):
    return math.sqrt(np.mean(np.square(y_pred-y_true)))/(np.mean(np.absolute(y_true)))

def generate_gaussian(mu, sigma):
    list_gauss = np.zeros(200,)
    list_gauss = random.gauss(mu, sigma)
    return np.mean(list_gauss)

def series_expand(series_input):
    return K.one_hot(K.cast(series_input[:,:,6],'int32'), num_series)

def output_expand(series_embed):
    series_embed = tf.expand_dims(series_embed, 1)
    return tf.concat([series_embed for i in range(window_size)], axis = 1)
    #return tf.reshape(out_layer, [-1, window_size, 2])

def train_model(window_size, LSTM_neurons):
    main_input = Input(shape=(None, 7), name = 'main_input')
    series_full = Lambda(series_expand, output_shape = (None, num_series))(main_input)
    series_embed = Dense(embed_size, activation = 'relu', kernel_initializer = 'truncated_normal',
    input_shape = (None, num_series))(series_full)
    x = Concatenate()([series_embed, main_input])
    x1 = LSTM(LSTM_neurons, dropout=0.4, kernel_initializer = 'truncated_normal', return_sequences = True)(x)
    x2 = LSTM(LSTM_neurons, dropout=0.2, kernel_initializer = 'truncated_normal', return_sequences = True)(x1)
    x3 = LSTM(LSTM_neurons, kernel_initializer = 'truncated_normal', return_sequences = True)(x2)
    param1 = TimeDistributed(Dense(1))(x3)
    param2 = TimeDistributed(Dense(1,activation='softplus'))(x3)
    output_layer = Concatenate(axis = 2)([param1, param2])
    model = Model(inputs=[main_input], outputs=[output_layer])
    print(model.summary())
    return model

name = "electricity_hourly"
window_size = 192
predict_size = 24
embed_size = 20
LSTM_neurons = 40
learning_rate = 0.001
batch_size = 64
nb_epoch = 2
num_series = 370
tot_epoch = 10
#preprocess(name+".csv", window_size, predict_size)
data_values = np.load(name+'_data.npy')
time_len = data_values.shape[0]
scale_values = np.load(name+'_scale.npy')
one_hot_labels = np.eye(num_series)
train_number = int(data_values.shape[0]*0.9)

train_model = train_model(window_size, LSTM_neurons)
adam = optimizers.Adam(lr=learning_rate)
train_model.compile(loss=log_likelihood, optimizer=adam)
chosen_batch = np.zeros(batch_size)
train_predict_y = np.zeros(time_len-train_number)
rmse_values = np.zeros(tot_epoch)
nd_values = np.zeros(tot_epoch)
print("Every epoch is defined as 100 batches of {}.".format(batch_size))
for i in range(1):
    print('======Epoch {} of {}======'.format(i, tot_epoch))
    for j in range(nb_epoch):
        chosen_batch = np.random.choice(np.arange(time_len), size = (batch_size,), p = scale_values[:, 1])
        print('batch {} out of {} has log-likelihood: {}'.format(j, nb_epoch, train_model.train_on_batch(
        [data_values[chosen_batch, :, 2:]], data_values[chosen_batch, :, 0:2])))
    chosen_batch = np.random.choice(np.arange(time_len), size = (1000,))
    predict_values = np.zeros((1000, window_size))
    predict_input = data_values[chosen_batch, :, 2:]
    for j in range(predict_size-1):
        predict_params = train_model.predict_on_batch(predict_input[:, :window_size-predict_size+j, :])
        predict_input[:, window_size-predict_size+j, 0] = predict_params[:, -1, 0]
    predict_params = train_model.predict_on_batch(predict_input)
    for j in range(1000):
        for k in range(window_size):
            predict_values[j,k] = predict_params[j,k,0]*scale_values[chosen_batch[j], 0]

    generate_values = np.zeros((1000, window_size))
    for j in range(1000):
        for k in range(window_size):
            generate_values[j, k] = generate_gaussian(predict_values[j, k], predict_params[j, k, 1]*math.sqrt(scale_values[chosen_batch[j], 0]))

    nd_values[i] = ND_metrics(data_values[chosen_batch,168:,0], predict_values[:, 168:])
    print("ND: ", nd_values[i])
    rmse_values[i] = RMSE_metrics(data_values[chosen_batch,168:,0], predict_values[:, 168:], 24)
    print("RMSE: ", rmse_values[i])

    x = np.arange(window_size)
    f = plt.figure()
    base = 8*100+10

    for k in range(8):
        plt.subplot(base+k+1)
        plt.plot(x, predict_values[k, :], color='b')
        plt.plot(x,data_values[chosen_batch[k], :, 0], color='r')
        plt.axvline(window_size-predict_size, color='g', linestyle = "dashed")

        #plt.pause(5)
    f.savefig("elect_simple_"+str(i)+".png")
    plt.close()
    x = np.arange(i+1)
    f = plt.figure()
    plt.plot(x, rmse_values[:i+1], color='b', label = 'RMSE_values')
    plt.plot(x, nd_values[:i+1], color='r', label = 'ND_values')
    plt.legend(loc='upper left')
    f.savefig("elect_simple_RMSE_ND_"+str(i)+".png")
    plt.close()
