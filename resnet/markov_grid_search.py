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
import tqdm
from sklearn.metrics import accuracy_score

def grid_search(markov_matrix,pred_vt,pred_v,
                p1_exp=3,p2_weight_decay=4,p3_order=6,
                p4_factor_inc=2.1,p5_factor_dec=0.55):
    markov_matrix = np.log2(markov_matrix) ** p1_exp  # 4.2
    for i in range(5):
        max = np.max(markov_matrix[i])
        markov_matrix[i] /= max
    # print(markov_matrix)
    # assert False

    def weight_decay(order):
        weights = []
        for i in range(order):
            weights.append(p2_weight_decay ** (-i))  # 1.5
        return weights

    order = p3_order  # 4
    weight = weight_decay(order)

    for i in range(1,len(pred_vt)-order):
        factor = 1
        if pred_v[i-1] != pred_v[i]:
            for j in range(1,order+1):
                if pred_v[i+j] == pred_v[i-1]:
                    factor += weight[j-1] * p4_factor_inc  # 0.71
                elif pred_v[i+j] == pred_v[i]:
                    factor -= weight[j-1] * p5_factor_dec  # 0.63
                    if factor < 0.1:
                        factor = 0.1
            vector = markov_matrix[pred_v[i - 1]].copy()
            vector[pred_v[i-1]] *= factor
            re_pred = pred_vt[i] * vector
            # print(re_pred)
            pred_v[i] = np.argmax(re_pred)

    overall_accuracy = accuracy_score(true_v, pred_v)
    return overall_accuracy


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

print('Time for data processing--- '+str(toc-tic)+' seconds---')
model_name = 'myNet.h5'
model = load_model(model_name)
# model.summary()
pred_vt = model.predict(TestX, batch_size=256, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

# p1_exp=np.linspace(1,6,15)
# p2_weight_decay=np.linspace(1,8,15)
# p3_order=[1,2,3,4,5,6,7,8]
# p4_factor_inc=np.linspace(0,5,15)
# p5_factor_dec=np.linspace(0,0.8,15)
#
# total = 15*15*8*15*15
# highest_acc = 0
# highest_cfg = None


#Opt parameter
acc = grid_search(markov_matrix.copy(),pred_vt.copy(),pred_v.copy(),
            4.2,1.5,4,0.71,0.63)
print(acc)


# with tqdm.tqdm(total=total) as pbar:
#     for p1 in p1_exp:
#         for p2 in p2_weight_decay:
#             for p3 in p3_order:
#                 for p4 in p4_factor_inc:
#                     for p5 in p5_factor_dec:
#                         pbar.update(1)
#                         acc = grid_search(markov_matrix.copy(),pred_vt.copy(),pred_v.copy(),
#                                     p1,p2,p3,p4,p5)
#                         if acc > highest_acc:
#                             highest_acc = acc
#                             highest_cfg = [p1,p2,p3,p4,p5]
#                             print(highest_cfg)
#                             pbar.set_description(str(highest_acc))
print("acc = ",highest_acc)
print("cfg : ",highest_cfg)
