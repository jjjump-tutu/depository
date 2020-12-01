import os
import numpy as np
from Config import Config
from Config import construct_config_18
import net
import deconv
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
parser.add_argument('-d', '--dataset', type=int, default=-1)
parser.add_argument('-g', '--gpu_id', type=str, default='0')
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
data1 = np.load('activation_8.npz')
testX1 = data1['testX']
testX1 = tf.convert_to_tensor(testX1)
data2 = np.load('activation_12.npz')
testX2 = data2['testX']
testX2 = tf.convert_to_tensor(testX2)
data3 = np.load('activation_16.npz')
testX3 = data3['testX']
testX3 = tf.convert_to_tensor(testX3)

origX, origY, orig_testX, orig_testY = dataload('channel0.npz')
print(np.var(origX))
print(np.mean(origX))
origX = tf.convert_to_tensor(origX)
origY = tf.convert_to_tensor(origY)
orig_testX = tf.convert_to_tensor(orig_testX)
orig_testY = tf.convert_to_tensor(orig_testY)

def pinyu(y):
    # 采样点数
    sampling_rate = 100  # 采样频率为8000Hz
    fft_size = 3072  # FFT处理的取样长度
    t = np.arange(0, 1.0, 1.0 / sampling_rate)  # np.arange(起点，终点，间隔)产生1s长的取样时间

    # N点FFT进行精确频谱分析的要求是N个取样点包含整数个取样对象的波形。因此N点FFT能够完美计算频谱对取样对象的要求是n*Fs/N（n*采样频率/FFT长度），
    # 因此对8KHZ和512点而言，完美采样对象的周期最小要求是8000/512=15.625HZ,所以156.25的n为10,234.375的n为15。
    xs = np.fft.fft(y) # 从波形数据中取样fft_size个点进行运算
    xf = np.fft.fftshift(xs) / fft_size   # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
    af =xf[xf.size // 2:]
    # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
    # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
    freqs = np.linspace(0, int(sampling_rate / 2), fft_size//2)
    # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # 在指定的间隔内返回均匀间隔的数字
    # xfp = 20 * np.log10(np.clip(np.abs(af), 1e-20, 1e100))
    xfp = np.clip(np.abs(af), 1e-20, 1e100)
    # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
    xfp = xfp[:int(xfp.shape[0]*0.4)]
    freqs = freqs[:int(freqs.shape[0]*0.4)]
    return freqs,xfp

    # pl.plot(f, absY)
    # pl.xlabel('freq(Hz)')
    # pl.title("fft")
    # pl.show()

for config in config_list:
    log_file = "deconv.log"
    print("Saving log as",log_file)
    log_templete = {"acc": None,
                    "epoch": None,
                    "cm": None,
                    "f1": None,
                    }
    # model = deconv.build_network(config)

    model_1 = os.path.join(model_dir,"deconv_ac8.h5")
    model_2 = os.path.join(model_dir, "deconv_ac12.h5")
    model_3 = os.path.join(model_dir, "deconv.h5")
    # checkpoint = ModelCheckpoint(filepath=model_name,
    #                              monitor='val_MSE', mode='min',
    #                              save_best_only='True')
    # lr_scheduler = LearningRateScheduler(config.lr_schedule)
    # callback_lists = [checkpoint,lr_scheduler]

    # log = model.fit(x=trainX, y=origX, batch_size=config.batch_size, epochs=arg.epoch,
    #           verbose=1, validation_data=(testX, orig_testX), callbacks=callback_lists)

    # acc_list = log.history["val_categorical_accuracy"]
    # highest_acc = max(acc_list)
    # highest_epoch = acc_list.index(max(acc_list))

    # del model
    import matplotlib.ticker as ticker
    model1 = load_model(model_1)
    pred_vt1 = model1.predict(testX1, batch_size=config.batch_size, verbose=1)
    model2 = load_model(model_2)
    pred_vt2 = model2.predict(testX2, batch_size=config.batch_size, verbose=1)
    model3 = load_model(model_3)
    pred_vt3 = model3.predict(testX3, batch_size=config.batch_size, verbose=1)
    t = np.arange(0, 30.72, 1/100)
    plt.figure(figsize = (25,10))
    ax = plt.subplot(421)
    plt.plot(t,orig_testX[0])
    plt.xlabel(u"Time(s)")
    # ax.xaxis.set_major_locator(ticker.NullLocator())  # 去掉横坐标值
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title('orignal signals')
    f, absY = pinyu(orig_testX[0])
    ax = plt.subplot(422)
    plt.plot(f,absY)
    plt.xlabel(u"Freq(Hz)")
    ax.set_title('spectrum graph')

    # ax.xaxis.set_major_locator(ticker.NullLocator())  # 去掉横坐标值
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax = plt.subplot(423)
    plt.plot(t,pred_vt1[0])
    plt.xlabel(u"Time(s)")
    # ax.xaxis.set_major_locator(ticker.NullLocator())  # 去掉横坐标值
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title('after the 5th residual block, MSE=0.0562')
    f, absY = pinyu(pred_vt1[0].squeeze(axis=-1))
    ax = plt.subplot(424)
    plt.plot(f,absY)
    plt.xlabel(u"Freq(Hz)")
    ax.set_title('spectrum graph')
    ax.yaxis.set_major_locator(ticker.NullLocator())


    ax = plt.subplot(425)
    plt.plot(t,pred_vt2[0])
    plt.xlabel(u"Time(s)")
    # ax.xaxis.set_major_locator(ticker.NullLocator())  # 去掉横坐标值
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title('after the 7th residual block, MSE=0.0817')
    f, absY = pinyu(pred_vt2[0].squeeze(axis=-1))
    ax = plt.subplot(426)
    plt.plot(f,absY)
    plt.xlabel(u"Freq(Hz)")
    ax.set_title('spectrum graph')
    ax.yaxis.set_major_locator(ticker.NullLocator())


    ax = plt.subplot(427)
    plt.plot(t,pred_vt3[0])
    plt.xlabel(u"Time(s)")
    # ax.xaxis.set_major_locator(ticker.NullLocator())  # 去掉横坐标值
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title('after the 9th residual block, MSE=0.1907')
    f, absY = pinyu(pred_vt3[0].squeeze(axis=-1))
    ax = plt.subplot(428)
    plt.plot(f,absY)
    plt.xlabel(u"Freq(Hz)")
    ax.set_title('spectrum graph')
    ax.yaxis.set_major_locator(ticker.NullLocator())

    plt.tight_layout()
    plt.savefig('compare_all.pdf')

    # pred_v = np.argmax(pred_vt, axis=1)
    # true_v = np.argmax(testY, axis=1)
    # cm = confusion_matrix(true_v, pred_v)
    # f1_macro = f1_score(true_v, pred_v, average='macro')


    # log_templete["acc"] = '{:.3%}'.format(highest_acc)
    # log_templete["epoch"] = str(highest_epoch)
    # log_templete["cm"] = str(cm)
    # log_templete["f1"] = str(f1_macro)
    # log = log_templete
    #
    # mail_context = ""
    #
    # with open(os.path.join(log_dir,log_file), mode="a") as f:
    #     f.write(log_file.split(".")[0])
    #     temp = "Highest acc = " + str(log_templete["acc"]) + "\n"
    #     mail_context += temp
    #     f.write(temp)
    #     temp = "Epoch = " + str(log_templete["epoch"]) + "\n"
    #     mail_context += temp
    #     f.write(temp)
    #     temp = "Cm = " + str(log_templete["cm"]) + "\n"
    #     mail_context += temp
    #     f.write(temp)
    #     temp = "F1 = " + str(log_templete["f1"]) + "\n"
    #     mail_context += temp
    #     f.write(temp)
    #
    # mail_subject = log_file.split(".")[0] + " Complete! Acc = " + str(log_templete["acc"])
    # mail_it("jiang_x@qq.com", mail_subject, mail_context)
    #
    # print("==========================================")
    # print(mail_context)