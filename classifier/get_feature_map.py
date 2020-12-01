# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:52:16 2019

@author: Winham

网络训练
"""

import os
import numpy as np
from Config import Config
# import net
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.models import load_model
# import mit_utils as utils
import time
import matplotlib.pyplot as plt

import tensorflow_addons as tfa
# from tqdm.keras import TqdmCallback
import tqdm
import tensorflow as tf
import datetime

from data_read import *
from data_preprocess import dataload

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
config = Config()
data_path = 'Data_npy/'

# --------------------- 数据载入和整理 -------------------------------

target_class = ['W', 'N1', 'N2', 'N3', 'REM']

# TrainX,TrainY,X_valid,Y_valid,TestX,TestY = loaddata('channel0.npz')
TrainX,TrainY,TestX,TestY = dataload('channel0.npz')
log_list = []

import time
log_file = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+".log"
print("Saving log as",log_file)
# print('Time for data processing--- '+str(toc-tic)+' seconds---')
#  assert False
# ------------------------ 网络生成与训练 ----------------------------

# Y_valid = np.vstack([Y_valid, TestY])

seq = 1
for i in range(seq):
    print(i,'-channel')
    # feature_map_model = Model(inputs=[model.input],outputs=[model.get_layer("feature_map").output])
    # feature_map_model.summary()
    # model.summary()
    # Train data
    # x = TrainX[:, :, i]
    # x = x[:,:,None]
    # # Val+Test data
    # X_v = X_valid[:,:,i]
    # X_v = X_v[:,:,None]
    # test = TestX[:,:,i]
    # test = test[:,:,None]
    # X_v = np.vstack([X_v,test])

    train_dataset = tf.data.Dataset.from_tensor_slices((TrainX, TrainY))
    test_dataset = tf.data.Dataset.from_tensor_slices((TestX, TestY))

    # del TrainX, TrainY, TestX, TestY

    model_name = 'AC-fpn.h5'
    model = load_model(model_name)
    # model.summary()
    feature_map_model = Model(inputs=[model.input],outputs=[model.get_layer("activation_17").output])

    feature_list = []
    lable_list = []

    for (x, y) in tqdm.tqdm(train_dataset.batch(config.batch_size)):
        feature = feature_map_model(x)
        feature =feature.numpy()
        feature_list.append(feature)
        lable_list.append(y.numpy())
    feature_list = np.concatenate(feature_list,axis=0)
    lable_list = np.concatenate(lable_list,axis=0)

    feature_list_test = []
    lable_list_test = []

    for (x, y) in tqdm.tqdm(test_dataset.batch(config.batch_size)):
        feature = feature_map_model(x)
        feature = feature.numpy()
        feature_list_test.append(feature)
        lable_list_test.append(y.numpy())
    feature_list_test = np.concatenate(feature_list_test, axis=0)
    lable_list_test = np.concatenate(lable_list_test, axis=0)




    with open('channel0-feature.npz', 'wb') as f:
        np.savez(
            f,
            X_train=feature_list,
            y_train=lable_list,
            X_test=feature_list_test,
            y_test=lable_list_test
        )

