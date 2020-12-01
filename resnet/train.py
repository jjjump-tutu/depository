# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:52:16 2019

@author: Winham

网络训练
"""

import os
import numpy as np
from Config import Config
import net
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
import mit_utils as utils
import time
import matplotlib.pyplot as plt
# from tqdm.keras import TqdmCallback
from data_preprocess import *

import tensorflow as tf
import datetime
import classifier
from data_read import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
config = Config()
data_path = 'Data_npy/'

target_class = ['W', 'N1', 'N2', 'N3', 'REM']

log_file = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+".log"
print("Saving log as",log_file)

tic = time.time()
# TrainX,TrainY,X_valid,Y_valid,TestX,TestY = loaddata('1.npz')
trainX, trainY, testX, testY = dataload('channel0_kk.npz')
print("Data time:", time.time() - tic)


# ------------------------ 网络生成与训练 ----------------------------
seq = 1
for i in range(seq):
    log_templete = {"channel": None,
                    "acc": None,
                    "epoch": None,
                    }
    print(i,'-channel')
    model = net.build_network(config)
    # model.summary()
    # model_name = 'bestbackbone.h5'
    # model = load_model(model_name)

    # feature_map_model = Model(inputs=[model.input],outputs=[model.get_layer("activation_16").output])

    #
    #
    # for layer in model.layers[:-5]:
    #     layer.trainable = False
    #
    #     # print('\n layers:', len(model.layers))
    #
    # net_arch = {'gap': True,
    #             'ap': False,
    #             'conv': [],
    #             'dropout': 0,
    #             'fc': [192, 96]}
    #
    # layer_name = ["dense", "dense_1", "dense_2"]
    # cla = classifier.build_classifier(config, net_arch)
    # for layer_n in layer_name:
    #     model.get_layer(layer_n).set_weights(cla.get_layer(layer_n).get_weights())
    #
    # net.add_compile(model, config)
    # print("trainable len:", len(model.trainable_variables))
    # 可训练层
    # for x in model.trainable_weights:
    #     print("tr:",x.name)
    # print('========================================')
    #
    # # 不可训练层
    # for x in model.non_trainable_weights:
    #     print("ntr:",x.name)
    # print('\n')
    # assert False



    model_name = 'myNet.h5'
    checkpoint = ModelCheckpoint(filepath=model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    log_dir = "logs/fit/" + str(i) + "-channel-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # dot_img_file = "model.png"
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    lr_scheduler = LearningRateScheduler(config.lr_schedule)
    callback_lists = [checkpoint, lr_scheduler]
    # x = TrainX[:, :, i]
    # x = x[:,:,None]
    # print(x.shape)
    # X_v = X_valid[:,:,i]
    # X_v = X_v[:,:,None]
    log = model.fit(x=trainX, y=trainY, batch_size=config.batch_size, epochs=config.train_epoch,
              verbose=2, validation_data=(testX, testY), callbacks=callback_lists)

    acc_list = log.history["val_categorical_accuracy"]
    highest_acc = max(acc_list)
    highest_epoch = acc_list.index(max(acc_list))

    # del TrainX, TrainY, TestX, TestY

    # model_name = 'myNet.h5'
    # model = load_model(model_name)
    # model.summary()
    # test = TestX[:,:,i]
    # test = test[:,:,None]
    # pred_vt = model.predict(test, batch_size=config.test_epoch, verbose=1)
    # # del TestX
    # pred_v = np.argmax(pred_vt, axis=1)
    # true_v = np.argmax(TestY, axis=1)
    #
    # utils.plot_confusion_matrix(true_v, pred_v, np.array(target_class))
    # utils.print_results(true_v, pred_v, target_class)
    # plt.show()
    log_templete["channel"] = str(i)
    log_templete["acc"] = '{:.3%}'.format(highest_acc)
    log_templete["epoch"] = str(highest_epoch)
    log = log_templete

    with open(log_file, mode="a") as f:
        temp = "========channel " + str(log_templete["channel"]) + "========" + "\n"
        f.write(temp)
        temp = "Highest acc = " + str(log_templete["acc"]) + "\n"
        f.write(temp)
        temp = "Epoch = " + str(log_templete["epoch"]) + "\n"
        f.write(temp)


