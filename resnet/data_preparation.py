# get 20-fold


import os
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random
from scipy import signal

seed = 0

np.random.seed(seed)


class Data_loader:
    X_seq = None
    y_seq = None
    X_seq_train = None
    y_seq_train = None
    X_seq_valid = None
    y_seq_valid = None
    X_seq_test = None
    y_seq_test = None

    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    def data_preparation(self, mypath="./data/eeg_fpz_cz"):
        # get file list
        file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        data_X, data_y = [], []
        for i in range(len(file_list)):
            with np.load(join(mypath, file_list[i])) as npz:
                # print(npz.files)
                data_X.append(npz['x'])
                data_y.append(npz['y'])
                # print(npz['fs'])

        # one-hot encoding sleep stages
        temp_y = []
        for i in range(len(data_y)):
            temp_ = []
            for j in range(len(data_y[i])):
                temp = np.zeros((5,))
                temp[data_y[i][j]] = 1.
                temp_.append(temp)
            temp_y.append(np.array(temp_))
        data_y = temp_y

        X_data = np.concatenate(data_X, axis=0).squeeze()
        y_data = np.concatenate(data_y, axis=0)
        print(X_data.shape)
        print(y_data.shape)
        #X_data = X_data[:10003]
        #y_data = y_data[:10003]

        #list_l = [0.5, 4, 8, 13, 30, 50, 70]
        list_l = [0.17, 4, 8, 13, 30]
        #list_h = [4, 8, 13, 30, 50, 70, 95]
        list_h = [4, 8, 13, 30, 49]
        #
        # train_data=np.array(data_X)
        # print(train_data.shape)

        result = np.zeros((X_data.shape[0], X_data.shape[1], 6))
        result[:, :, 0] = X_data.copy()

        for i in range(len(list_l)):
            print("Filter ",i)
            w1 = 2 * list_l[i] / 100
            w2 = 2 * list_h[i] / 100
            b, a = signal.butter(4, [w1, w2], btype='bandpass')  # 配置滤波器 8 表示滤波器的阶数
            result[:, :, i + 1] = signal.filtfilt(b, a, X_data)  # data为要过滤的信号

        # data_sum = np.sum(result[:, :, 1:], axis=-1)
        # print(np.mean(abs(result[:, :, 0] - data_sum)))
        # print(np.mean(abs(result[:, :, 0])))
        # print(np.mean(abs(data_sum)))

        # import matplotlib.pyplot as plt
        # for i in range(result.shape[0]):
        #     for j in range(5):
        #         subplt = 81*10+j+1
        #         plt.subplot(subplt)
        #         sp = np.fft.fft(result[i,:,j+1])
        #         freq = np.fft.fftfreq(result[i,:,j+1].shape[-1],d=0.01)
        #         plt.plot(freq, sp)

                #plt.title("Band "+str(j))
            # plt.subplot(817)
            # plt.plot(np.sum(result[i,:,1:],axis=-1))
            # #plt.title("Reconstructed")
            # plt.subplot(818)
            # plt.plot(np.sum(result[i,:,1:],axis=-1)-result[i,:,0])
        #     #plt.title("Reconstruct Error")
        #     plt.show()
        #     input()
        # assert False

        # make sequence data
        # seq_length = 25  # 30s*25

        # X_seq, y_seq = [], []
        # res = [result]
        #
        # for i in range(len(res)):
        #     for j in range(0, len(res[i]), seq_length):  # discard last short sequence
        #         if j + seq_length < len(res[i]):
        #             X_seq.append(np.array(res[i][j:j + seq_length]))
        #             y_seq.append(np.array(res[i][j:j + seq_length]))
        # X_seq = np.array(X_seq)
        # y_seq = np.array(y_seq)
        trun_length = int(result.shape[0]-result.shape[0]%25)
        X_seq = result[:trun_length,].reshape(result.shape[0]//25,25,3000,6)
        y_seq = y_data[:trun_length,].reshape(result.shape[0]//25,25,5)

        # X_data = np.concatenate(data_X,axis=0)
        # y_data = np.concatenate(data_y,axis=0)
        # #
        print(X_seq.shape)
        print(y_seq.shape)
        # print(X_data.shape)
        # print(y_data.shape)
        # (3229, 25, 3000, 1)
        # (3229, 25, 5)
        # (71,)
        # (71,)

        self.X_seq = X_seq
        self.y_seq = y_seq

    def rotate(self, l, k):
        n = int(len(l) * 1 / 20 * k)
        l = l[-n:] + l[:-n]
        return l

    def get_k_th_seq(self, X_seq, y_seq, k):
        seq_idx = [i for i in range(len(X_seq))]
        random.shuffle(seq_idx)
        seq_idx = self.rotate(seq_idx, k)

        idx_train = int(len(X_seq) * 0.8)
        idx_valid = int(len(X_seq) * 0.1) + 1
        idx_test = int(len(X_seq) * 0.1) + 1

        X_seq_train, y_seq_train = [], []
        X_seq_valid, y_seq_valid = [], []
        X_seq_test, y_seq_test = [], []

        for i in range(0, idx_train):
            idx = seq_idx[i]
            X_seq_train.append(X_seq[idx])
            y_seq_train.append(y_seq[idx])

        for i in range(idx_train, idx_train + idx_valid):
            idx = seq_idx[i]
            X_seq_valid.append(X_seq[idx])
            y_seq_valid.append(y_seq[idx])

        for i in range(idx_train + idx_valid, len(seq_idx)):
            idx = seq_idx[i]
            X_seq_test.append(X_seq[idx])
            y_seq_test.append(y_seq[idx])

        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)

        X_seq_valid = np.array(X_seq_valid)
        y_seq_valid = np.array(y_seq_valid)

        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)

        self.X_seq_train = X_seq_train
        self.y_seq_train = y_seq_train
        self.X_seq_valid = X_seq_valid
        self.y_seq_valid = y_seq_valid
        self.X_seq_test = X_seq_test
        self.y_seq_test = y_seq_test

    # This method should follow right after get_k_th_seq
    def get_k_th_data(self, X_seq_train, y_seq_train, X_seq_valid, y_seq_valid, X_seq_test, y_seq_test):
        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []

        for i in range(len(X_seq_train)):
            for j in range(len(X_seq_train[i])):
                X_train.append(X_seq_train[i][j])
                y_train.append(y_seq_train[i][j])

        for i in range(len(X_seq_valid)):
            for j in range(len(X_seq_valid[i])):
                X_valid.append(X_seq_valid[i][j])
                y_valid.append(y_seq_valid[i][j])

        for i in range(len(X_seq_test)):
            for j in range(len(X_seq_test[i])):
                X_test.append(X_seq_test[i][j])
                y_test.append(y_seq_test[i][j])

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_valid = np.array(X_valid)
        self.y_valid = np.array(y_valid)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

    # def bandpass(self,X_seq,y_seq):
    #     list_l = [0.5, 4, 8, 13, 30, 50, 70]
    #     list_h = [4, 8, 13, 30, 50, 70, 95]
    #     #
    #     # train_data=np.array(data_X)
    #     # print(train_data.shape)
    #
    #     for i in range(len(list_l)):
    #         w1 = 2 * list_l[i] / 95
    #         w2 = 2 * list_h[i] / 95
    #         b, a = signal.butter(8, [w1, w2], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
    #         train_data[:, :, i + 1] = signal.filtfilt(b, a, data_X)  # data为要过滤的信号


if __name__ == "__main__":
    d = Data_loader()
    d.data_preparation()
    # assert False

    path = './20_fold_data'
    if os.path.exists(path) is False:
        os.mkdir(path)

    i=1

    d.get_k_th_seq(d.X_seq, d.y_seq, i)
    d.get_k_th_data(d.X_seq_train, d.y_seq_train, d.X_seq_valid, d.y_seq_valid, d.X_seq_test, d.y_seq_test)
    with open(join(str(i) + '.npz'), 'wb') as f:
        np.savez(
            f,
            X_seq_train=d.X_seq_train,
            y_seq_train=d.y_seq_train,
            X_seq_valid=d.X_seq_valid,
            y_seq_valid=d.y_seq_valid,
            X_seq_test=d.X_seq_test,
            y_seq_test=d.y_seq_test,
            X_train=d.X_train,
            y_train=d.y_train,
            X_valid=d.X_valid,
            y_valid=d.y_valid,
            X_test=d.X_test,
            y_test=d.y_test
        )


    print("done")


