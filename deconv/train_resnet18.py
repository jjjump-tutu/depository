import os
import numpy as np
from Config import Config
from Config import construct_config_18
import net
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import mit_utils as utils
import time
import matplotlib.pyplot as plt
# from tqdm.keras import TqdmCallback
from data_preprocess import *

import tensorflow as tf
import datetime
import argparse
from mail import mail_it


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', type=int)
parser.add_argument('-g', '--gpu_id', type=str, default=0)
parser.add_argument('-l', '--length', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=60)


arg = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id

config_list = construct_config_18(arg.length)

target_class = ['W', 'N1', 'N2', 'N3', 'REM']
log_dir = "./logs/"
model_dir = "./weights/"
os.makedirs(log_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)

dataset_name = os.path.join('k-fold','channel0_{}_fold.npz'.format(arg.dataset))
trainX, trainY, testX, testY = dataload('channel0.npz')
trainX = tf.convert_to_tensor(trainX)
trainY = tf.convert_to_tensor(trainY)
testX = tf.convert_to_tensor(testX)
testY = tf.convert_to_tensor(testY)

for config in config_list:
    log_file = "ResNet{}_{}_{}fold.log".format(config.depth, config.conv_filter_length,arg.dataset)
    print("Saving log as",log_file)
    log_templete = {"acc": None,
                    "epoch": None,
                    "cm": None,
                    "f1": None,
                    }
    # model = net.build_network(config)
    model = load_model('myNet.h5')

    model_name = os.path.join(model_dir,"ResNet{}_{}_{}fold.log".format(config.depth, config.conv_filter_length,arg.dataset))
    checkpoint = ModelCheckpoint(filepath=model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    lr_scheduler = LearningRateScheduler(config.lr_schedule)
    callback_lists = [checkpoint, lr_scheduler]

    log = model.fit(x=trainX, y=trainY, batch_size=config.batch_size, epochs=arg.epoch,
              verbose=2, validation_data=(testX, testY), callbacks=callback_lists)

    acc_list = log.history["val_categorical_accuracy"]
    highest_acc = max(acc_list)
    highest_epoch = acc_list.index(max(acc_list))

    del model

    model = load_model(model_name)
    pred_vt = model.predict(testX, batch_size=config.batch_size, verbose=1)
    pred_v = np.argmax(pred_vt, axis=1)
    true_v = np.argmax(testY, axis=1)
    cm = confusion_matrix(true_v, pred_v)
    f1_macro = f1_score(true_v, pred_v, average='macro')


    log_templete["acc"] = '{:.3%}'.format(highest_acc)
    log_templete["epoch"] = str(highest_epoch)
    log_templete["cm"] = str(cm)
    log_templete["f1"] = str(f1_macro)
    log = log_templete

    mail_context = ""

    with open(os.path.join(log_dir,log_file), mode="a") as f:
        f.write(log_file.split(".")[0])
        temp = "Highest acc = " + str(log_templete["acc"]) + "\n"
        mail_context += temp
        f.write(temp)
        temp = "Epoch = " + str(log_templete["epoch"]) + "\n"
        mail_context += temp
        f.write(temp)
        temp = "Cm = " + str(log_templete["cm"]) + "\n"
        mail_context += temp
        f.write(temp)
        temp = "F1 = " + str(log_templete["f1"]) + "\n"
        mail_context += temp
        f.write(temp)

    mail_subject = log_file.split(".")[0] + " Complete! Acc = " + str(log_templete["acc"])
    mail_it("jiang_x@qq.com", mail_subject, mail_context)

    print("==========================================")
    print(mail_context)