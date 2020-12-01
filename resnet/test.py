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
import argparse
from mail import mail_it
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', type=int, default=0)
parser.add_argument('-g', '--gpu_id', type=str, default='0')
# parser.add_argument('-l', '--length', type=int, default=32)
# parser.add_argument('-e', '--epoch', type=int, default=60)


arg = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id

target_class = ['W', 'N1', 'N2', 'N3', 'REM']
log_dir = "./logs/"
model_dir = "./weights/back/"
os.makedirs(log_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)
if arg.dataset == -1:
    dataset_name = 'channel0.npz'
else:
    dataset_name = os.path.join('k-fold','channel0_{}_fold.npz'.format(arg.dataset))
    # dataset_name = os.path.join('k-fold','data20_{}_fold.npz'.format(arg.dataset))
    # dataset_name = os.path.join('k-fold','data_pzoz_20_{}_fold.npz'.format(arg.dataset))

target_class = ['W', 'N1', 'N2', 'N3', 'REM']
target_sig_length = 3072
tic = time.time()
trainX, trainY, TestX, TestY = dataload(dataset_name)
toc = time.time()

log_file = "{}_vis.log".format(arg.dataset)
print("Saving log as",log_file)
log_templete = {"acc": None,
                    "cm": None,
                    "f1": None,
                "per F1":None,
                    }

markov_matrix = [[66927., 3996., 179., 6., 86.],
                 [2252., 17891., 4269., 9., 753.],
                 [1271., 2262., 80861., 3546., 1043.],
                 [179., 113., 3247., 15892., 23.],
                 [565., 912., 427., 1., 32279.]]
# markov_matrix = [[4642.,  226. ,  30.,    2.   , 8.],
#  [ 101.,  816. , 318.  ,  1.,   51.],
#  [ 104.,  161. ,7752. , 410. , 107.],
#  [  20. ,  11. , 381. ,2151. ,   1.],
#  [  41. ,  73. ,  53.  ,  0. ,3207.],]

markov_matrix = np.array(markov_matrix)

# markov_matrix_copy = markov_matrix.copy()
# for i in range(5):
#     markov_matrix_copy[i] /= markov_matrix_copy[i].sum()
# print(markov_matrix_copy)
markov_matrix = np.log2(markov_matrix) ** 4.2
# markov_matrix = markov_matrix ** 0.5
for i in range(5):
    max = np.max(markov_matrix[i])
    markov_matrix[i] /= max
# print(markov_matrix)
# assert False

print('Time for data processing--- '+str(toc-tic)+' seconds---')
# model_name = os.path.join(model_dir,'fpn_net_18_3w_ac_{}fold_.h5'.format(arg.dataset))
model_name = 'myNet.h5'
model = load_model(model_name)
# model.summary()
pred_vt = model.predict(TestX, batch_size=256, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

orig = pred_v.copy()




def weight_decay(order):
    weights = []
    for i in range(order):
        weights.append(1.5 ** (-i))
    return weights

order = 4
weight = weight_decay(order)

for i in range(1,len(pred_vt)-order):
    factor = 1
    if pred_v[i-1] != pred_v[i]:
        for j in range(1,order+1):
            if pred_v[i+j] == pred_v[i-1]:
                factor += weight[j-1]*0.71
            elif pred_v[i+j] == pred_v[i]:
                factor -= 0.63 * weight[j-1]
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
highest_acc = utils.print_results(true_v, pred_v, target_class)
plt.savefig('cm_{}fold_vis.png'.format(arg.dataset))




def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)

cm = confusion_matrix(true_v, pred_v)
f1_macro = f1_score(true_v, pred_v, average='macro')

i=0
f1 = []
for i in range(5):
    print(i ,':')
    r = calculate_label_recall(cm,i)
    p = calculate_label_prediction(cm,i)
    f = calculate_f1(p,r)
    f1.append(f)



log_templete["acc"] = '{:.3%}'.format(highest_acc)
log_templete["cm"] = str(cm)
log_templete["f1"] = str(f1_macro)
log_templete["per F1"] = str(f1)
log = log_templete

mail_context = ""

with open(os.path.join(log_dir, log_file), mode="a") as f:
    f.write(log_file.split(".")[0])
    temp = "Highest acc = " + str(log_templete["acc"]) + "\n"
    mail_context += temp
    f.write(temp)
    temp = "Cm = " + str(log_templete["cm"]) + "\n"
    mail_context += temp
    f.write(temp)
    temp = "F1 = " + str(log_templete["f1"]) + "\n"
    mail_context += temp
    temp = "per F1 = " + str(log_templete["per F1"]) + "\n"
    mail_context += temp
    f.write(temp)

mail_subject = log_file.split(".")[0] + " Complete! Acc = " + str(log_templete["acc"])
mail_it("jiang_x@qq.com", mail_subject, mail_context)

print("==========================================")
print(mail_context)

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

import matplotlib.ticker as ticker
orig = orig[4500:5400]
orig.resize((30,30))
# plt.subplot(131)
plt.matshow(orig, cmap = plt.cm.Blues)
# ax1.xaxis.set_major_locator(ticker.NullLocator())   #去掉横坐标值
# ax1.yaxis.set_major_locator(ticker.NullLocator())  #去掉纵坐标值
plt.xticks([])  #去掉横坐标值
plt.yticks([])
# ax1.set_title('orig')

plt.savefig('cm_orig_pred.pdf')

print(pred_v.shape)
pred_v = pred_v[4500:5400]
pred_v.resize((30,30))
# plt.subplot(132)

plt.matshow(pred_v, cmap = plt.cm.Blues)
# ax2.xaxis.set_major_locator(ticker.NullLocator())   #去掉横坐标值
# ax2.yaxis.set_major_locator(ticker.NullLocator())
plt.xticks([])  #去掉横坐标值
plt.yticks([])
# ax2.set_title('pred')

plt.savefig('cm_pred.pdf')

true_v = true_v[4500:5400]
true_v.resize((30,30))
# plt.subplot(133)
plt.matshow(true_v, cmap = plt.cm.Blues)
# ax3.set_title('true')
# ax3.xticks([])  #去掉横坐标值
# ax3.yticks([])  #去掉纵坐标值
 #去掉纵坐标值
plt.xticks([])  #去掉横坐标值
plt.yticks([])
plt.savefig('cm_true.pdf')
# plt.tight_layout()
# plt.savefig('compare.pdf')

