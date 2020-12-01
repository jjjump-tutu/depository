
import os
import numpy as np
from Classifier_config import Config
# import net
import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
# import mit_utils as utils
import time
import matplotlib.pyplot as plt
# from tqdm.keras import TqdmCallback
from resample import resample
from option import Options
from classifier import build_classifier

import tensorflow as tf
import datetime

from data_read import *
from keras.optimizers import SGD


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
config = Config()

opt = Options().getparse()
#=====================================================================
def train_classifier(dataset, net_arch=None, config=None,log='',count=None):
    assert len(dataset)==4,"Expected length of dataset is 4, but got " + str(len(dataset)) + "!"
    trainX,trainY,testX,testY = dataset

    log_dir = "./logs/" + log + "/"
    os.makedirs(log_dir, exist_ok=True)

    print('=============================================')
    print(net_arch)

    log_file = log_dir + str(count) + "_cls.log"
    print("Saving log as", log_file)

    # ------------------------ 网络生成与训练 ----------------------------

    seq = 1
    for i in range(seq):
        log_templete = {"channel": None,
                        "acc": None,
                        "epoch": None,
                        "rebalanced": None,
                        "net_arch":None,
                        }
        print(i, '-channel')

        model = build_classifier(config,net_arch)

        # dot_img_file = log_dir + str(count) + ".png"
        # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

        optimizer = SGD(lr=config.lr_schedule(0), momentum=0.9)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        # model_name = log_dir + 'best_classifier.h5'
        # checkpoint = ModelCheckpoint(filepath=model_name,
        #                              monitor='val_categorical_accuracy', mode='max',
        #                              save_best_only='True')

        # log_dir = "logs/fit/" + str(i) + "-channel-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        lr_scheduler = LearningRateScheduler(config.lr_schedule)
        # callback_lists = [checkpoint, lr_scheduler, tensorboard_callback]
        # callback_lists = [checkpoint, lr_scheduler]
        callback_lists = [lr_scheduler]

        log = model.fit(x=trainX, y=trainY, batch_size=config.batch_size, epochs=config.train_epoch,
                        verbose=2, validation_data=(testX, testY), callbacks=callback_lists)
        acc_list = log.history["val_categorical_accuracy"]
        highest_acc = max(acc_list)
        highest_epoch = acc_list.index(max(acc_list))

        # del TrainX, TrainY, TestX, TestY

        log_templete["channel"] = str(i)
        log_templete["acc"] = '{:.3%}'.format(highest_acc)
        log_templete["epoch"] = str(highest_epoch)
        log_templete["rebalanced"] = opt.rebalanced
        log_templete["net_arch"] = str(net_arch)
        log = log_templete

        with open(log_file, mode="a") as f:
            temp = "========channel " + str(log_templete["channel"]) + "========" + "\n"
            f.write(temp)
            temp = "Highest acc = " + str(log_templete["acc"]) + "\n"
            f.write(temp)
            temp = "Epoch = " + str(log_templete["epoch"]) + "\n"
            f.write(temp)
            temp = "rebalanced:\n" + str(log_templete["rebalanced"]) + "\n"
            f.write(temp)
            temp = "net_arch:\n" + str(log_templete["net_arch"]) + "\n"
            f.write(temp)
    return highest_acc, count, net_arch



# --------------------- 数据载入和整理 -------------------------------

target_class = ['W', 'N1', 'N2', 'N3', 'REM']

npz = np.load('channel0-feature.npz')
trainX = npz['X_train']
trainY = npz['y_train']
testX = npz['X_test']
testY = npz['y_test']
# print(trainX.shape)
# assert False

if opt.rebalanced:
    trainX, trainY = resample(trainX, trainY, mode='max')

net_arch_list = []

for gap in [False]:
    for ap in [2,3,4,6,8,16]:
        for conv in [[]]:
        #  for conv in [[[48, 3]], [[48, 3], [12, 3]], [[96, 12], [48, 12], [12, 12]]]:
            for dropout in [0.3, 0.5]:
                for fc in [[50, 30], [80, 50, 30]]:
    # for ap in [2,3,4,6,8]:
    #     for conv in [[]]:
    #     # for conv in [[[48, 3]], [[48, 3], [12, 3]], [[96, 12], [48, 12], [12, 12]]]:
    #         for dropout in [0.5]:
    #             for fc in [[192,96]]:
                    net_arch_list.append({'gap': gap,
                                          'ap': ap,
                                          'conv': conv,
                                          'dropout': dropout,
                                          'fc': fc})
log_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
count = 1
res_list=[]
for net_arch in net_arch_list:
    acc, num, net = train_classifier([trainX,trainY,testX,testY],net_arch,config,log_path,str(count))
    count += 1
    res_list.append({'acc': acc, 'count': num, 'net_arch': net})

res_list.sort(key=lambda x:x['acc'], reverse=True)
log_file = "./logs/"+ log_path +"/summary_cls.log"
best_result = res_list[0]
with open(log_file, mode="w") as f:
    temp = "Highest acc = " + str(best_result['acc']) + "\n"
    f.write(temp)
    temp = "count = " + str(best_result['count']) + "\n"
    f.write(temp)
    temp = "net_arch:\n" + str(best_result["net_arch"]) + "\n"
    f.write(temp)
