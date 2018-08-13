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
import random
import math
#import tf.contrib.distributions.NormalWithSoftplusScale as NORM


# load dataset
#data = np.load('reframed-data-1000.npy')
#data = np.load('reframed-data-10000.npy')
#data = np.load('reframed-data-19999.npy')
data = np.load('reframed-data-all.npy')
#data = data[200000:300000, :, :]
v_i = np.load('vi-g-all.npy')
#v_i = v_i[200000:300000, :]
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
N = ((int(n_samples * 2 / 3)) // 64) * 64 # number of samples in train data

# do the scaling
#ground_truth = data[:,:,0]
#local_age = data[:,:,1]  % window_length

# v_i = np.zeros((data.shape[0],data.shape[1], 1))

#v_i = np.asarray([ [ np.mean([data[i,j,0]]) for j in range(data.shape[1]) ] for i in range(data.shape[0])]).reshape(-1, window_length, 1) + 1

#def expand():
#    output = np.zeros((n_samples,window_length,n_dims))
#    n_samples_each_f = n_samples // n_dims
#    for i in range(n_dims):
#        output[i*n_samples_each_f: (i+1)*n_samples_each_f,:,i] = 1
#    return output


#def gaussian(x, mean, std):
#    print("\n***in gaussian****")
#    print("x.shape: ", x.shape)
#    print("mean.shape: ", mean.shape)
#    print('std.shape: ', std.shape)
#    a = K.exp(tf.divide(-K.square(x-mean),(K.constant(2)*K.square(std))))
#    b = K.sqrt(K.constant(2*math.pi)*K.square(std))
#    likelihood = tf.divide(a,b)
#    print("likelihood.shape: ",likelihood.shape)
#    print("***end of gaussian****\n")
#    return likelihood

def log_gaussian(x, mean, std):
    dist = tf.contrib.distributions.NormalWithSoftplusScale(mean,std)
    likelihood = dist.log_prob(x)
    return likelihood

def sum_log_likelihood(y_true, para_pred):
    print('\n==in custom loss===')
    print("y_true.shape: ", y_true.shape)
    print("para_pred.shape: ", para_pred.shape)
    mean = para_pred[:,:,0]
    mean = tf.expand_dims(mean, axis = 2)
    std = (para_pred[:,:,1])
    std = tf.expand_dims(std, axis = 2)
    print("mean.shape: ", mean.shape)
    print('std.shape: ', std.shape)
    likelihood = log_gaussian(y_true, mean, std)
    print("likelihood.shape: ",likelihood.shape)
    print('==end of custom loss===')
    result =  K.mean(likelihood)
    return -result

#aux_in = Input(shape=(input_window_length,n_dims, ), name='aux_input')
aux_in = Input(shape=(None, ), name='aux_input', dtype='int32')

# in salute to https://gist.github.com/bzamecnik/a33052ec46ee7efeb217856d98a4fb5f
aux_in_full = Lambda(K.one_hot, arguments={'num_classes': n_dims}, output_shape=(None, n_dims))(aux_in)
x = Dense(20)(aux_in_full)

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

# cannot use Flatten
#print("mean_for_each.shape:")
#print(mean_for_each.shape)
#mean_for_each = GlobalAveragePooling1D()(mean_for_each)
#mean = Dense(1)(mean_for_each)
#print("mean.shape:")
#print(mean.shape)
# alternaative:
# https://stackoverflow.com/questions/47795697/how-to-give-variable-size-images-as-input-in-keras

std_for_each = TimeDistributed(Dense(1, activation='softplus'))(lstm_out3)
#print("std_for_each.shape:")
#print(std_for_each.shape)
#std_for_each = GlobalAveragePooling1D()(std_for_each)
#std = Dense(1)(std_for_each)
#print("std.shape:")
#print(std.shape)
out_for_each = layers.concatenate([mean_for_each,std_for_each],axis = 2)
#out = layers.concatenate([mean,std])
model = Model(inputs=[aux_in,main_in], outputs=[out_for_each])
#model = Model(inputs=[aux_in,main_in], outputs=[out])
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
    # mean = tf.expand_dims(mean, axis = 1)
    # print("in nd: mean.shape: ", mean.shape)
    #print("mean.shape", mean.shape)
    #print("y_true.shape", y_true.shape)
    #print("vi.shape", vi.shape)
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
#test_aux_input = (np.arange(n_dims) == test_aux_input[...,None]-1).astype(int)
test_vi = v_i[N:, :]
rewritten_input = np.copy(test_main_input)
batch_size = 64
n_batch = (n_samples - N) // batch_size

nd = np.zeros(n_batch)
rmse = np.zeros(n_batch)

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg') # local
matplotlib.use('Agg') # server


def plot(label, prediction,num_plot,k,e): # 1,192  1,192  1,192
    x = np.arange(192)
    f = plt.figure()
    base = num_plot*100+10
    for i in range(num_plot):
        label_temp = label[i].reshape([window_length,])
        pred_temp = prediction[i].reshape([window_length,])
        plt.subplot(base+i+1)
        plt.plot(x,label_temp, color='b')
        plt.plot(x,pred_temp, color='r')
        plt.axvline(168, color='k', linestyle = "dashed")
		#参考线
    #plt.pause(5)
#    plot.show()
    f.savefig(str(e)+'thEpoch'+str(k)+"thBatch.png")
    plt.close()

n_epoches = 10
for e in range(n_epoches):
    model.fit([train_aux_input,train_main_input], [train_y] , epochs=1, batch_size=64,verbose=1)
    for i in random.sample(range(n_batch), 10):
        #print('******\n******\n******\nbatch number: ', i)
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
            #========== get prediction for next sequence =================
            rewritten_input[batch_range, input_window_length + j, 0] = pred_result[:, -1 ,0]
            
        plot(test_main_input[i*batch_size:i*batch_size+8,:,0], rewritten_input[i*batch_size:i*batch_size+8,:,0],8,i,e)
        
        nd[i] = nd_metrics(test_main_input[batch_range, input_window_length:, 0], rewritten_input[batch_range, input_window_length:, 0], test_vi[batch_range])
        rmse[i] = rmse_metrics(test_main_input[batch_range, input_window_length:, 0], rewritten_input[batch_range, input_window_length:, 0], test_vi[batch_range])
        
        print('batch: ', i)
        print('nd: ', nd[i])
        print('rmse: ', rmse[i])
    
    np.save(str(e)+'thEpochOfND.npy', nd)
    np.save(str(e)+'thEpochOfRMSE.npy', rmse)
     


#''' # <---------------- corresponds to structure check
