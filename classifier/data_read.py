import numpy as np
import h5py
#
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def onehot(data_y):
    # one-hot encoding sleep stages
    data_y=[data_y]
    temp_y = []
    for i in range(len(data_y)):
        temp_ = []
        for j in range(len(data_y[i])):
            temp = np.zeros((6,))
            temp[data_y[i][j]] = 1.
            temp_.append(temp)
        temp_y.append(np.array(temp_))
    data_y = np.array(temp_y)
    return data_y

np.set_printoptions(suppress=True)
def loaddata(dataset):
    npz = np.load(dataset)
    trainX = npz['X_train']
    trainY = npz['y_train']
    X_valid = npz['X_valid']
    y_valid = npz['y_valid']
    testX = npz['X_test']
    testY = npz['y_test']

    # trainX=np.array(data["train_data"]).astype('float32').transpose(0,2,1)
    # trainY=np.array(data["train_label"]).astype('int').transpose(1,0)
    # testX=np.array(data["test_data"]).astype('float32').transpose(0,2,1)
    # testY = np.array(data["test_label"]).astype('int').transpose(1,0)

    # X = np.zeros((trainX.shape[0]+testX.shape[0],trainX.shape[1],trainX.shape[2]))
    # X[0:trainX.shape[0],:,:] = trainX
    # X[trainX.shape[0]:trainX.shape[0]+testX.shape[0],:,:]=testX
    #
    # Y = np.append(trainY,testY)
    #
    # result=[X,Y]
    # trainY = onehot(trainY).transpose(1,2,0).squeeze()
    # testY = onehot(testY).transpose(1,2,0).squeeze()

    temp_x = np.zeros((trainX.shape[0],3072,trainX.shape[2]))
    temp_x[:,36:3036,:]=trainX
    trainX = temp_x

    temp_x = np.zeros((X_valid.shape[0],3072,trainX.shape[2]))
    temp_x[:,36:3036,:]=X_valid
    X_valid = temp_x

    temp_x = np.zeros((testX.shape[0],3072,testX.shape[2]))
    temp_x[:,36:3036,:]=testX
    testX = temp_x

    print(trainX.shape)
    print(trainY.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(testX.shape)
    print(testY.shape)


    result = [trainX,trainY,X_valid,y_valid,testX,testY]
    return result
#
# trainX,trainY=loaddata('sleep_edf_data.mat')
#
# #_ = [print(len(x)) for x in loaddata('sleep_edf_data.mat')]
# # print(type(testX))
# # print(testX.shape)
# # print(trainX.shape)
# print(trainX.shape)
# print(trainY)


