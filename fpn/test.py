# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:2 7:28 2019

@author: Winham

网络测试
"""

import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from data_preprocess import *
import mit_utils as utils
import time
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


target_class = ['W', 'N1', 'N2', 'N3', 'REM']
target_sig_length = 3072
tic = time.time()
trainX, trainY, TestX, TestY = dataload('channel0.npz')
toc = time.time()

markov_matrix = [[66927., 3996., 179., 6., 86.],
                 [2252., 17891., 4269., 9., 753.],
                 [1271., 2262., 80861., 3546., 1043.],
                 [179., 113., 3247., 15892., 23.],
                 [565., 912., 427., 1., 32279.]]

markov_matrix = np.array(markov_matrix)

# markov_matrix_copy = markov_matrix.copy()
# for i in range(5):
#     markov_matrix_copy[i] /= markov_matrix_copy[i].sum()
# print(markov_matrix_copy)
markov_matrix = np.log2(markov_matrix) ** 3
for i in range(5):
    max = np.max(markov_matrix[i])
    markov_matrix[i] /= max
# print(markov_matrix)
# assert False

print('Time for data processing--- '+str(toc-tic)+' seconds---')
model_name = 'myNet.h5'
model = load_model(model_name)
# model.summary()
pred_vt = model.predict(TestX, batch_size=256, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

def weight_decay(order):
    weights = []
    for i in range(order):
        weights.append(4 ** (-i))
    return weights

order = 6
weight = weight_decay(order)

for i in range(1,len(pred_vt)-order):
    factor = 1
    if pred_v[i-1] != pred_v[i]:
        for j in range(1,order+1):
            if pred_v[i+j] == pred_v[i-1]:
                factor += weight[j-1]*2.1
            elif pred_v[i+j] == pred_v[i]:
                factor -= 0.55 * weight[j-1]
                if factor < 0.1:
                    factor = 0.1
        vector = markov_matrix[pred_v[i - 1]].copy()
        vector[pred_v[i-1]] *= factor
        re_pred = pred_vt[i] * vector
        # print(re_pred)
        pred_v[i] = np.argmax(re_pred)



# f1 = 3.1
# f2 = 0.45
# for i in range(1,len(pred_vt)-1):
#     if pred_v[i-1] != pred_v[i]:
#         if pred_v[i-1] == pred_v[i+1]:
#             factor = f1
#         elif pred_v[i] == pred_v[i+1]:
#             factor = f2
#         else:
#             factor = 1
#             # print(pred_vt[i])
#         vector = markov_matrix[pred_v[i - 1]].copy()
#         vector[pred_v[i-1]] *= factor
#         re_pred = pred_vt[i] * vector
#         # print(re_pred)
#         pred_v[i] = np.argmax(re_pred)


utils.plot_confusion_matrix(true_v, pred_v, np.array(target_class))
utils.print_results(true_v, pred_v, target_class)
plt.savefig('cm.png')

# pred_v = pred_v[:10000]
# pred_v.resize((100,100))
# plt.subplot(121)
# plt.matshow(pred_v, cmap = plt.cm.Blues)
# plt.savefig('cm_pred.png')
#
# true_v = true_v[:10000]
# true_v.resize((100,100))
# plt.subplot(122)
# plt.matshow(true_v, cmap = plt.cm.Blues)
# plt.savefig('cm_true.png')
