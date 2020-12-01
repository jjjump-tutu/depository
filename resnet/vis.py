
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图，来源：
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    # if not title:
    #     if normalize:
    #         title = 'Normalized confusion matrix'
    #     else:
    #         title = 'Confusion matrix, without normalization'



    # classes = classes[unique_labels(y_true, y_pred)]
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm)

    fig, ax = plt.subplots()
    cm = np.array(cm)
    # for i in range(5):
    #     cm[i,i] = 0
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Source stage',
           xlabel='Target stage')

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # fmt = '.2f'
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j],fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig('data_v.pdf')
    return ax

target_class = ['W', 'N1', 'N2', 'N3', 'REM']
markov_matrix = [[60846, 3612,  183,    9,   83,],
 [ 2042, 16099, 3837,    11, 661],
 [ 1166, 2030,72165, 3136,  970],
 [  163,  102, 2875, 14339,   21],
 [  516,  807,  407,     5, 28945],]
markov_matrix = np.array(markov_matrix)
# cm = np.zeros((5,5))
# all = []
# for i in range(5):
#     all = markov_matrix.sum(axis=1)[i]
#     print(all)
#     for j in range(5):
#         cm[i,j] = markov_matrix[i,j] / all
#         print(cm[i,j])

# markov_matrix = markov_matrix.astype('float') / markov_matrix.sum(axis=1)[:, np.newaxis]
# print(markov_matrix)
plot_confusion_matrix(markov_matrix, np.array(target_class))