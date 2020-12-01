from data_read import *
import numpy as np
import matplotlib.pyplot as plt
import time
import operator as op
from data_preprocess import dataload
tic = time.time()
# TrainX,TrainY,X_valid,Y_valid,TestX,TestY = loaddata('2.npz')
# data = np.load('data_pzoz_20.npz')
trainX, trainY, testX, testY = dataload('./k-fold/channel0_3_fold.npz')
# x = data['x']
# y = data['y']
# print(x.shape)
# print(y.shape)
print (trainY.shape)
print("Data time:", time.time() - tic)
# assert False
y =np.argmax(trainY, axis=1)
print (y.shape)
# y = np.argmax(y,axis = 1)
# plt.figure()
# for i in range(8):
#     plt.subplot(520+i+1)
#     plt.plot(x[i])
P = np.zeros((5,5))
for i in range(y.shape[0]):
    P[y[i-1],y[i]] += 1
# for i in range(5):
#     P[i,i] = 0
print(P)
# plt.matshow(P, cmap = plt.cm.Blues)

# y = y[:10000]
# y.resize((100,100))
# plt.matshow(y, cmap = plt.cm.Blues)
# plt.savefig("vis.png")


#
# count = {}
# for i in y[0:len(y) - 1]:
#     count[i] = count.get(i, 0) + 1
# count = sorted(count.items(), key=op.itemgetter(0), reverse=False)
#
# markov_marix = np.zeros([len(count), len(count)])
# for j in range(len(y) - 1):
#     for m in range(len(count)):
#         for n in range(len(count)):
#             if y[j] == count[m][0] and y[j + 1] == count[n][0]:
#                 markov_marix[m][n] += 1
# for t in range(len(count)):
#     markov_marix[t, :] /= count[t][1]
# print(markov_marix)
