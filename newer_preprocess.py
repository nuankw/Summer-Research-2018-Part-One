# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:03:25 2018

@author: Nuan Wen
"""

from pandas import read_csv
import numpy as np

input_window_length = 168
output_window_length = 24
window_length = input_window_length + output_window_length
n_i = 370
stride_size = 24

def reframe(data, input_win_size, output_win_size, stride_size, n_i):
    assert (data.shape[1] == n_i)
    
    # how many windows for one series
    n_windows = (data.shape[0] - input_win_size) // stride_size
    window_size = output_win_size + input_win_size
    n_covariates = 3
    
    # with one-hot
    # output = np.zeros((n_windows * n_i, window_size, 1 + n_covariates + n_i))
    # without one-hot
    output = np.zeros((n_windows * n_i, window_size, 1 + n_covariates + 1), dtype = int)
    
    local_age = np.array([x for x in range(window_size)])
    hour_of_day = local_age % 24
    day_of_week = local_age // 24
    
    # go through one feature over entire series first
    
    for i in range(n_i):
        # for embedding: 
        embed_indicator = np.zeros((n_windows, window_size))
        embed_indicator.fill(i+1)
        output[i*n_windows : (i+1) * n_windows,:,4] = embed_indicator
        # ground truth and covariate
        # all features share same time-dependent covariate AGE at same time_stamp
        for j in range(n_windows):
            #print('feature: ' + str(i))
            #print('window: ' + str(j))
            ground_truth = data[j*stride_size:j*stride_size + window_size,i]
            #print(ground_truth)
            #print(ground_truth.shape)
            output[i*n_windows + j,:,0] = ground_truth
            output[i*n_windows + j,:,1] = local_age + j * window_size # age
            output[i*n_windows + j,:,2] = hour_of_day # hour of day
            output[i*n_windows + j,:,3] = day_of_week  # day of week
            
            
    return output
''' comment this line if want to add one-hot embedding to preprocess
        # each feature has a item-dependent one-hot representation
        one_hot = np.zeros((,))
        one_hot[i] = 1
        for k in range(window_size):
            output[i * n_windows + k, :, 4 : 4 + n_i] = one_hot
            
    # return output, one_hot
    return output
#'''         
# preprocess
#dataset = read_csv('first1000.csv', header=0, index_col=0)
#dataset = read_csv('first10000.csv', header=0, index_col=0)
#dataset = read_csv('first19999.csv', header=0, index_col=0)
dataset = read_csv('electricity_hourly.csv', header=0, index_col=0)
dataset.fillna(0, inplace=True)
values = dataset.values
# ensure all data is float
# values = values.astype('float32')
# frame as supervised learning
reframed = reframe(values, input_window_length, output_window_length, stride_size, n_i)
print(reframed.shape)
print(np.sum(reframed[:,:,0]))
v_i = np.asarray([ [np.mean(reframed[i,:,0]) + 1] for i in range(reframed.shape[0])])
np.save('vi-all.npy', v_i)
#np.save('vi-19999.npy', v_i)
#np.save('vi-10000.npy', v_i)
#np.save('vi-1000.npy', v_i)
reframed[:,:,0] = reframed[:,:,0] / v_i
print(np.sum(reframed[:,:,0]))
''' no longer needed, but a fancy method
# inspired by https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
my_permu = generate_permutation_order(n_i, n_i*(input_window_length + output_window_length))
i = np.argsort(my_permu)
reframed = reframed.values[:,i]
reframed = np.array(np.hsplit(reframed,n_i))
print(reframed.shape)
chosen_list = [ (stride_size * x) for x in range(reframed.shape[1]//stride_size + 1)]
reframed = reframed[:, chosen_list ,:]
'''
#np.save('reframed-data-1000.npy', reframed)
#np.save('reframed-data-10000.npy', reframed)
#np.save('reframed-data-19999.npy', reframed)
np.save('reframed-data-all.npy', reframed)
