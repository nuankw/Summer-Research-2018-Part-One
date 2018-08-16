# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:39:14 2018

@author: Nuan Wen
"""

# switched to functional model for easier modification
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, LSTM, Dropout, TimeDistributed, Lambda, Flatten, GlobalAveragePooling1D
from keras import layers
import keras.backend as K
import tensorflow as tf
from keras import optimizers
#import random
import math
#import tf.contrib.distributions.NormalWithSoftplusScale as NORM


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg') # local
#matplotlib.use('Agg') # server


def plot(label,prediction,vi, num_plot,k,e): # 1,192  1,192  1,192
    x = np.arange(192)
    f = plt.figure()
    base = num_plot*100+10
    for i in range(num_plot):
        label_temp = label[i].reshape([window_length,]) * vi[i]
        pred_temp = prediction[i].reshape([window_length,]) * vi[i]
        plt.subplot(base+i+1)
        plt.plot(x,label_temp, color='b')
        plt.plot(x,pred_temp, color='r')        
        plt.axvline(168, color='k')
    #plt.pause(5)
#    plot.show()
    print('saving...')
    f.savefig(str(e)+'thEpoch'+str(k)+"thBatch.png")
    plt.close()


# load dataset
#data = np.load('reframed-data-1000.npy')
#data = np.load('reframed-data-10000.npy')
#data = np.load('reframed-data-19999.npy')
data = np.load('reframed-data-all.npy')
#data = data[0:10000, :, :]
v_i = np.load('vi-g-all.npy')
#v_i = v_i[0:10000, :]
print("data.shape: ", data.shape)
print("v_i.shape: ", v_i.shape)

# get the dimension
input_window_length = 168
output_window_length = 24
window_length = input_window_length + output_window_length
n_features = 4
n_dims = 370
input_embed_dim = 370
output_embed_dim = 20
n_samples = data.shape[0]
N = ((int(n_samples * 0.9)) // 64) * 64 # number of samples in train data

def log_gaussian(x, mean, std):
    dist = tf.contrib.distributions.NormalWithSoftplusScale(mean,std)
    likelihood = dist.log_prob(x)
    return likelihood

def sum_log_likelihood(y_true, para_pred):
    print('\n==in custom loss===')
    print("y_true.shape: ", y_true.shape)
    print("para_pred.shape: ", para_pred.shape)
    mean = para_pred[:,:,0]
    std = (para_pred[:,:,1])
    print("mean.shape: ", mean.shape)
    print('std.shape: ', std.shape)
    likelihood = log_gaussian(y_true[:,:,0], mean, std)
    print("likelihood.shape: ",likelihood.shape)
    print('==end of custom loss===')
    result =  K.mean(likelihood)
    return -result

#aux_in = Input(shape=(input_window_length,n_dims, ), name='aux_input')
aux_in = Input(shape=(None, ), name='aux_input', dtype='int32')

# in salute to https://gist.github.com/bzamecnik/a33052ec46ee7efeb217856d98a4fb5f
aux_in_full = Lambda(K.one_hot, arguments={'num_classes': n_dims}, output_shape=(None, n_dims))(aux_in)
x = Dense(20, activation='sigmoid')(aux_in_full)

#x = Embedding(input_dim=370, output_dim=20, input_length = 192)(aux_in)
main_in = Input(shape=(None, n_features, ), name="main_input")
input1 = layers.concatenate([main_in, x])
lstm_out1 = LSTM(40, return_sequences = True)(input1)
drop_out1 = Dropout(0.2)(lstm_out1)
lstm_out2 = LSTM(40,  return_sequences = True)(drop_out1)
drop_out2 = Dropout(0.2)(lstm_out2)
lstm_out3 = LSTM(40,  return_sequences = True)(lstm_out2)
drop_out3 = Dropout(0.2)(lstm_out3)
mean_for_each = TimeDistributed(Dense(1))(drop_out3)
std_for_each = TimeDistributed(Dense(1, activation='softplus'))(lstm_out3)
out_for_each = layers.concatenate([mean_for_each,std_for_each],axis = 2)
model = Model(inputs=[aux_in,main_in], outputs=[out_for_each])
adam = optimizers.Adam(lr=0.01)
model.compile(loss=sum_log_likelihood, optimizer=adam)
print(model.summary())

#''' ----------------> uncomment this line to just print model
# train
# train set
train_main_input = data[:N,:, 0:4] # ground truth and covariates
train_aux_input =  np.array(data[:N,:,4], dtype='int32') # the one-hot position
#train_aux_input = (np.arange(n_dims) == train_aux_input[...,None]-1).astype(np.int32, copy=False)
train_y = data[:N,:,5].reshape(-1, window_length, 1)

print(train_main_input.shape, train_aux_input.shape)

#model.fit([train_aux_input,train_main_input], [train_y] , epochs=1, batch_size=64,verbose=1)

'''''' #----------------> edit this line to train or test model
# ============================================================
# predict

def rmse_metrics(y_true, mean, vi):
    y_true = y_true * vi
    mean = mean * vi
    denom = np.mean(np.absolute(y_true))
    if (denom == 0.0):
        denom = -1.0
    return math.sqrt(np.mean(np.square(mean-y_true)))/denom

def nd_metrics(y_true, mean, vi):
    y_true = y_true * vi
    mean = mean * vi
    denom = np.sum(np.absolute(y_true))
    if (denom == 0.0):
        denom = -1.0
    return np.sum(np.absolute(mean-y_true))/denom


test_main_input = data[N:,:,0:4] # ground truth and covariates
test_aux_input = np.array(data[N:,:,4], dtype='int32') # the one-hot position
test_vi = v_i[N:, :]
rewritten_input = np.copy(test_main_input)
batch_size = 64
n_batch = (n_samples - N) // batch_size

nd = np.zeros(n_batch)
rmse = np.zeros(n_batch)

n_epoches = 6
#selection = random.sample(range(n_batch), 5)
for e in range(n_epoches):
    print('epoch: ',e+1, '/', n_epoches)
    model.fit([train_aux_input,train_main_input], [train_y] , epochs=1, batch_size=32,verbose=1, shuffle=True)
    model.save_weights(str(e)+'th_Epoch_weights.h5')
    for i in range(n_batch):
        batch_range = range(i*batch_size, (i+1)*batch_size)
        for j in range(output_window_length): # j = [0.23], input_window_length+j = [168,191]
            #=========== index, input, dimension check ====================
            #print('\n========= now predicting the index (start from 0): ',input_window_length+j, '=========')
            #print('which means that the current input is: [0,' + str(input_window_length+j-1) + ']')
            #print('aka. ":'+str(input_window_length+j)+'"')
            main_input = rewritten_input[batch_range,:input_window_length + j,0:4]
            #print('so the main input for this round of prediction is:\n', main_input)
            #print('shape of the main input for this round of prediction is: ', main_input.shape) 
            # from (64,168,4) to (64,191,4)
            aux_input = test_aux_input[batch_range,:input_window_length + j]
            #print('shape of the auxiliary input for this round of prediction is: ', aux_input.shape) 
            # from (64,168) to (64,191)
            #=========== make the prediction ==============================
            pred_result = model.predict_on_batch([aux_input, main_input])
            #========== get prediction for next sequence ==================
            rewritten_input[batch_range, input_window_length + j, 0] = pred_result[:, -1 ,0]
            #========== draw the prediction ===============================
        
        nd[i] = nd_metrics(test_main_input[batch_range, input_window_length:, 0], rewritten_input[batch_range, input_window_length:, 0], test_vi[batch_range])
        rmse[i] = rmse_metrics(test_main_input[batch_range, input_window_length:, 0], rewritten_input[batch_range, input_window_length:, 0], test_vi[batch_range])
        if (nd[i] <= 0.1 | rmse[i] <= 0.5):
            plot(test_main_input[i*batch_size:i*batch_size+8,:,0], rewritten_input[i*batch_size:i*batch_size+8,:,0], test_vi[i*batch_size:i*batch_size+8], 8,i,e)
    
    print('nd average: ', np.mean(nd))
    print('rmse average: ', np.mean(rmse))
    
    print('nd min: ', np.min(nd))
    print('rmse min: ', np.min(rmse))
    
    print('nd max: ', np.max(nd))
    print('rmse max: ', np.max(rmse))
    
    np.save(str(e)+'thEpochOfND.npy', nd)
    np.save(str(e)+'thEpochOfRMSE.npy', rmse)


#''' # <---------------- corresponds to structure check
