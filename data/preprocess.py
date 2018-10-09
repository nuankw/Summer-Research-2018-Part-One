"""
Created on Wed Jul 25 14:03:25 2018

@author: Nuan Wen
"""

from pandas import read_csv
import numpy as np

input_window_length = 168
output_window_length = 24
window_length = input_window_length + output_window_length
n_features = 370
stride_size = 24


def reframe(data, input_win_size, output_win_size, stride_size, n_features):
    assert (data.shape[1] == n_features)

    # how many windows for one series, -1 for the shifted ground truth (label)
    n_windows = (data.shape[0] - input_win_size - 1) // stride_size
    # the size of each window
    window_size = output_win_size + input_win_size

    # output a reframed dataset, with n_windows windows, each of size window_size
    output = np.zeros((n_windows * n_features, window_size, 1 + 3 + 1 + 1), dtype='float32')
    # 1: ground truth
    # 3: covarates
    # 1: embedding
    # 1: shifted groud truth (label)

    local_age = np.array([x for x in range(window_size)])
    hour_of_day = local_age % 24
    hour_of_day = (hour_of_day - np.mean(hour_of_day)) / np.std(hour_of_day)
    # print('hour of day: ', hour_of_day)
    day_of_week = local_age // 24
    day_of_week = (day_of_week - np.mean(day_of_week)) / np.std(day_of_week)
    # print('day_of_week: ', day_of_week)

    # go through one feature over entire series first
    n_all_zero_windows = 0
    for i in range(n_features):
        # fill in 1), 2) & 3):
        # ground truth, covariates and label
        # all features share same time-dependent covariate AGE at same time_stamp
        for j in range(n_windows-1):
            ground_truth = data[j*stride_size:j*stride_size + window_size, i]
            # if the entire window is all zero, ignore it (don't include it in the output)
            if (np.sum(ground_truth) == 0):
                n_all_zero_windows = n_all_zero_windows + 1
                continue
            ground_truth_shifted = data[j*stride_size+1:j*stride_size + window_size+1, i]
            output[i*n_windows + j, :, 0] = ground_truth
            age = local_age + j * window_size
            output[i*n_windows + j, :, 1] = age  # age

            output[i*n_windows + j, :, 2] = hour_of_day  # hour of day
            output[i*n_windows + j, :, 3] = day_of_week  # day of week
            output[i*n_windows + j, :, 5] = ground_truth_shifted  # y
        # for embedding:
        embed_indicator = np.zeros((n_windows, window_size))
        embed_indicator.fill(i+1)
        output[i*n_windows: (i+1) * n_windows, :, 4] = embed_indicator

    output[:, :, 1] = (output[:, :, 1] - np.mean(output[:, :, 1])) / np.std(output[:, :, 1])

    return output[:(-1*n_all_zero_windows), :, :]


if (__name__ == '__main__'):
    # preprocess
    raw_files = ['trainRaw.csv', 'testRaw.csv']
    reframed_files = ['trainReframed.npy', 'testReframed.npy']
    vi_files = ['trainVi.npy', 'testVi.npy']
    for i in range(2):
        raw_file = raw_files[i]
        dataset = read_csv(raw_file, header=0, index_col=0)
        dataset.fillna(0, inplace=True)
        values = dataset.values
        print(values.shape)

        reframed = reframe(values,
                           input_window_length,
                           output_window_length,
                           stride_size,
                           n_features).astype('float32')
        print(reframed.shape)

        v_i = np.asarray([[np.mean(reframed[i, :, 0]) + 1] for i in range(reframed.shape[0])])
        reframed[:, :, 0] = (reframed[:, :, 0] / v_i)
        reframed[:, :, 5] = (reframed[:, :, 5] / v_i)
        assert reframed.shape[0] == v_i.shape[0]
        order = np.arange(0, reframed.shape[0], 1)
        np.random.shuffle(order)
        reframed = reframed[order, :, :]
        v_i = v_i[order, :]

        reframed_file = reframed_files[i]
        vi_file = vi_files[i]
        np.save(vi_file, v_i)
        np.save(reframed_file, reframed)

'''
# no longer needed, but a fancy method
# inspired by https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
my_permu = generate_permutation_order(n_features, n_features*(input_window_length + output_window_length))
i = np.argsort(my_permu)
reframed = reframed.values[:, i]
reframed = np.array(np.hsplit(reframed, n_features))
print(reframed.shape)
chosen_list = [(stride_size * x) for x in range(reframed.shape[1]//stride_size + 1)]
reframed = reframed[:, chosen_list, :]
'''
