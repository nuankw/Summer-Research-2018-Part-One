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

    # with one-hot
    # output = np.zeros((n_windows * n_i, window_size, 1 + n_covariates + n_i))
    # without one-hot
    output = np.zeros((n_windows * n_i, window_size, 1 + 3 + 1 + 1), dtype = 'float32')
    # 1: ground truth
    # 3: covarates
    # 1: embedding
    # 1: shifted groud truth

    local_age = np.array([x for x in range(window_size)])
    hour_of_day = local_age % 24
    hour_of_day = (hour_of_day - np.mean(hour_of_day)) / np.std(hour_of_day)
    #print('hour of day: ', hour_of_day)
    day_of_week = local_age // 24
    day_of_week = (day_of_week - np.mean(day_of_week)) / np.std(day_of_week)
    #print('day_of_week: ', day_of_week)

    # go through one feature over entire series first

    for i in range(n_i):
        # for embedding:
        embed_indicator = np.zeros((n_windows, window_size))
        embed_indicator.fill(i+1)
        output[i*n_windows : (i+1) * n_windows,:,4] = embed_indicator
        # ground truth and covariate
        # all features share same time-dependent covariate AGE at same time_stamp
        for j in range(n_windows-1):
            #print('feature: ' + str(i))
            #print('window: ' + str(j))
            ground_truth = data[j*stride_size:j*stride_size + window_size,i]
            ground_truth_shifted = data[j*stride_size+1:j*stride_size + window_size+1,i]
            #print(ground_truth)
            #print(ground_truth.shape)
            #print(ground_truth_shifted.shape)
            output[i*n_windows + j,:,0] = ground_truth
            age = local_age + j * window_size
            output[i*n_windows + j,:,1] = age # age

            output[i*n_windows + j,:,2] = hour_of_day # hour of day
            output[i*n_windows + j,:,3] = day_of_week  # day of week
            output[i*n_windows + j,:,5] = ground_truth_shifted # y
    output[:,:,1] = (output[:,:,1] - np.mean(output[:,:,1])) / np.std(output[:,:,1])

    print("output:")
    print(output)
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
#values = values.astype('float32')
# frame as supervised learning
reframed = reframe(values, input_window_length, output_window_length, stride_size, n_i).astype('float32')
print(reframed.shape)
print(np.sum(reframed[:,:,0]))
v_i_g = np.asarray([ [np.mean(reframed[i,:,0]) + 1] for i in range(reframed.shape[0])])
reframed[:,:,0] = (reframed[:,:,0] / v_i_g)
reframed[:,:,5] = (reframed[:,:,5] / v_i_g)
assert reframed.shape[0] == v_i_g.shape[0]
order = np.arange(0,reframed.shape[0],1)
np.random.shuffle(order)
reframed = reframed[order,:,:]
v_i_g = v_i_g[order,:]
# v_i_s = np.asarray([ [np.mean(reframed[i,:,5]) + 1] for i in range(reframed.shape[0])])
np.save('vi-g-all.npy', v_i_g)
#np.save('vi-19999.npy', v_i)
#np.save('vi-10000.npy', v_i)
#np.save('vi-1000.npy', v_i)
# print("reframed before: ", reframed[-10:, 0:5, 0])

#print("reframed after: ", reframed[-10:, 0:5, 0])
#print("v_i: ", v_i[-10:])
#print(np.sum(reframed[:,:,0]))
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
print(reframed)
np.save('reframed-data-all.npy', reframed)
