import numpy as np

from keras.utils import to_categorical


def preprocess(dataset):
    data  = np.load(dataset)
    x = data['x']
    y = data['y']
    print(x.shape)
    print(y.shape)
    sc = 152
    st = 44

    train_length = int(0.7*x.shape[0]/ (sc+st)) - 1
    test_length = int(0.3 * x.shape[0] / (sc+st)) - 1
    segment_length = train_length + test_length

    trainX = []
    trainY = []
    testX = []
    testY = []

    # ==========================每人73分=====================================
    # i = 0
    # while i < x.shape[0]:
    #     if i + segment_length >= x.shape[0]:
    #         break
    #     trainX.append(x[i:i + train_length])
    #     trainY.append(y[i: i + train_length])
    #     testX.append(x[i + train_length: i + segment_length])
    #     testY.append(y[i + train_length: i + segment_length])
    #     i += segment_length
    #=============================人群73分===================================
    length = x.shape[0]
    sc = sc / (sc + st)
    st = st / (sc + st)
    i = 0
    sc_train = int(sc * 0.7 * length)
    sc_test = int(sc * 0.3 * length)
    st_train = int(st * 0.7 * length)
    st_test = int(st * 0.3 * length)

    trainX.append(x[i: sc_train])
    trainY.append(y[i: sc_train])
    testX.append(x[sc_train: sc_train + sc_test])
    testY.append(y[sc_train: sc_train + sc_test])

    i = sc_train + sc_test

    trainX.append(x[i: i + st_train])
    trainY.append(y[i: i + st_train])
    testX.append(x[i + st_train: i + st_train + st_test])
    testY.append(y[i + st_train: i + st_train + st_test])





    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)
    testX = np.concatenate(testX)
    testY = np.concatenate(testY)
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    filename = 'channel0_kk.npz'
    np.savez(filename,
             trainX = trainX,
             trainY = trainY,
             testX = testX,
             testY = testY
             )


def dataload(dataset):
    data = np.load(dataset)
    trainX = data['trainX']
    trainY = data['trainY']
    testX = data['testX']
    testY = data['testY']

    temp_x = np.zeros((trainX.shape[0],3072))
    temp_x[:,36:3036]=trainX
    trainX = temp_x

    temp_x = np.zeros((testX.shape[0],3072))
    temp_x[:,36:3036]=testX
    testX = temp_x

    result = [trainX, trainY, testX, testY]
    return result



if __name__ == '__main__':
    preprocess('data.npz')




