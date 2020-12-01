import numpy as np
from keras.utils import to_categorical
import math


def resample(x, y_onehot,mode='avg'):
    y = np.argmax(y_onehot,axis=1)
    idx = y.argsort()
    class_counter_sum = [0]
    class_counter = []
    num_class = y.max() + 1
    for i in range(num_class):
        c = np.sum(y == i)
        class_counter_sum.append(c + class_counter_sum[i])
        class_counter.append(c)
    print(class_counter)
    # y_sort = y[idx]
    x_sort = x[idx]
    sliced_x = []
    for i in range(num_class):
        sliced_x.append(x_sort[class_counter_sum[i]:class_counter_sum[i+1]])
    if mode == 'avg':
        sample_num = math.ceil(y.shape[0] / num_class)
    elif mode == 'min':
        sample_num = min(class_counter)
    elif mode == 'max':
        sample_num = max(class_counter)
    else:
        sample_num = -1
    rebalanced_x = []
    for i in range(num_class):
        if class_counter[i] >= sample_num:
            oversample = False
        else:
            oversample = True
        x_idx = np.random.choice(class_counter[i], sample_num, replace=oversample)
        rebalanced_x.append(sliced_x[i][x_idx])
        print(len(rebalanced_x[i]))
    x = np.concatenate(rebalanced_x, axis=0)
    y = np.array([_ for _ in range(num_class)] * sample_num)
    y.sort()
    y_onehot = to_categorical(y)
    return x, y_onehot
