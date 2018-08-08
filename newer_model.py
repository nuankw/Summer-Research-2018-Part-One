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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import math
#import tf.contrib.distributions.NormalWithSoftplusScale as NORM


# load dataset
#data = np.load('reframed-data-1000.npy')
#data = np.load('reframed-data-10000.npy')
#data = np.load('reframed-data-19999.npy')
data = np.load('reframed-data-all.npy')
data = data[90000:100000, :, :]
v_i = np.load('vi-all.npy')
v_i = v_i[90000:100000, :]
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

def neg_log_gaussian(x, mean, std):
    dist = tf.contrib.distributions.NormalWithSoftplusScale(mean,std)
    likelihood = tf.scalar_mul(-1, dist.log_prob(x))
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
    likelihood = neg_log_gaussian(y_true, mean, std)
    print("likelihood.shape: ",likelihood.shape)
    print('==end of custom loss===')
    return K.mean(likelihood)

#aux_in = Input(shape=(input_window_length,n_dims, ), name='aux_input')
aux_in = Input(shape=(None, ), name='aux_input', dtype='int32')

# in salute to https://gist.github.com/bzamecnik/a33052ec46ee7efeb217856d98a4fb5f
aux_in_full = Lambda(K.one_hot, arguments={'num_classes': n_dims}, output_shape=(None, n_dims))(aux_in)
x = Dense(20)(aux_in_full)

#x = Embedding(input_dim=370, output_dim=20, input_length = 192)(aux_in)
main_in = Input(shape=(None, n_features, ), name="main_input")
input1 = layers.concatenate([main_in, x])
lstm_out1 = LSTM(40, return_sequences = True)(input1)
drop_out1 = Dropout(0.4)(lstm_out1)
lstm_out2 = LSTM(40,  return_sequences = True)(drop_out1)
drop_out2 = Dropout(0.2)(lstm_out2)
lstm_out3 = LSTM(40,  return_sequences = True)(lstm_out2)

mean_for_each = TimeDistributed(Dense(1))(lstm_out3)

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
out_for_each = layers.concatenate([mean_for_each,std_for_each])
#out = layers.concatenate([mean,std])
model = Model(inputs=[aux_in,main_in], outputs=[out_for_each])
#model = Model(inputs=[aux_in,main_in], outputs=[out])
adam = optimizers.Adam(lr=0.01)
model.compile(loss=sum_log_likelihood, optimizer=adam)
print(model.summary())

#''' ----------------> uncomment this line to just print model
# train
# train set
train_main_input = data[:N,:-1, 0:4] # ground truth and covariates
train_aux_input =  np.array(data[:N,:-1,4], dtype='int32') # the one-hot position
#train_aux_input = (np.arange(n_dims) == train_aux_input[...,None]-1).astype(np.int32, copy=False)
train_y = data[:N,1:,0].reshape(-1, window_length, 1)

print("---train_main_input:---")
print(train_main_input[-10:, :5, 0])
print("---train_y:---")
print(train_main_input[-10:, :5, 0])
print('====== train data: ======')
print(train_main_input.shape, train_aux_input.shape)

model.fit([train_aux_input,train_main_input], [train_y] , epochs=1, batch_size=64,verbose=1, shuffle=True)

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


test_main_input = data[N:,:-1,0:4] # ground truth and covariates
test_aux_input = np.array(data[N:,:-1,4], dtype='int32') # the one-hot position
#test_aux_input = (np.arange(n_dims) == test_aux_input[...,None]-1).astype(int)
test_y = data[N:,1:,0].reshape(-1, window_length, 1)
test_pred = np.copy(test_main_input)
print('====== test data: ======')
print(test_main_input.shape, test_aux_input.shape)
test_vi = v_i[N:, :]
batch_size = 64
n_batch = (n_samples - N) // batch_size
nd = np.zeros(n_batch)
rmse = np.zeros(n_batch)
for i in range( n_batch ):
#for i in range( 100,101 ): # just for test
    print('batch number: ', i+1)
    for j in range(output_window_length+1):
        #print("prediction round: (of 24) ", j+1)
        this_batch_predict = model.predict([test_aux_input[i*64:(i+1)*64,:input_window_length + j], test_pred[i*64:(i+1)*64,:input_window_length + j,0:4]],batch_size = 64, verbose=0)
        this_batch_predict = np.asarray(this_batch_predict)
        #print("this_batch_predict.shape", this_batch_predict.shape)
        this_batch_mean = this_batch_predict[:,:,0]
        test_pred[i*64:(i+1)*64,:input_window_length + j, 0] = this_batch_mean
        #print("this_batch_predict.shape", this_batch_predict.shape)
        #print(this_batch_predict)
    nd[i] = nd_metrics(test_main_input[i*64:(i+1)*64, :, 0], test_pred[i*64:(i+1)*64, :, 0], test_vi[i*64:(i+1)*64])
    rmse[i] = rmse_metrics(test_main_input[i*64:(i+1)*64, :, 0], test_pred[i*64:(i+1)*64, :, 0], test_vi[i*64:(i+1)*64])
    print('nd[batch_number]: ', nd[i])
    print('rmse[batch_number]: ', rmse[i])
    #this_batch_score = model.evaluate([test_aux_input[i*64:(i+1)*64,:,:input_window_length], test_main_input[i*64:(i+1)*64,:,:input_window_length]],this_batch_predict , batch_size = 64 , verbose=1)
    #print(this_batch_score)
    #print('Test loss:', this_batch_score[0])
    #print('Test accuracy:', this_batch_score[1])
#print(parahat.shape)
#score =
#''' # <---------------- corresponds to structure check
