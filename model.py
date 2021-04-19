#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:19:55 2021

@author: jiuqiwang
"""
import random
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
import tensorflow as tf

# Temple resizing function
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# utils.py
###################################################################################


# Rescale to be 64 frames
def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new


def sampling_frame(p, C):
    full_l = p.shape[0]  # full length
    if random.uniform(0, 1) < 0.5:  # aligment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l  # sample end point
        p = p[int(s):int(e), :, :]
    else:  # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom(p, C.frame_l, C.joint_n, C.joint_d)
    return p


from scipy.spatial.distance import cdist


def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        # distance max
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M


def normlize_range(p):
    # normolize to start point, use the center for hand case
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(8, 8)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%.1f' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                # annot[i, j] = '%.1f%%\n%d' % (p, c)
                annot[i, j] = '%.1f' % (p)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=False, cmap="YlGnBu")
    plt.savefig(filename)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(123)


class Config:
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 12  # the number of joints
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_coarse = 14  # the number of coarse class
        self.clc_fine = 28  # the number of fine-grained class
        self.feat_d = 231
        self.filters = 64  # ?? In the python notebook is 64, here is 16...
        self.data_dir = 'SHREC'
C = Config()

def poses_diff(x):
    H, W = x.get_shape()[1], x.get_shape()[2]
    x = tf.subtract(x[:, 1:, ...], x[:, :-1, ...])  # tensorflow method
    x = tf.image.resize(x, size=[H, W])
    return x


# calculate M_k_slow and M_k_fast
def pose_motion(P, frame_l):
    print("P type is:")
    print(type(P))
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l, -1))(P_diff_slow)  # M_slow_[1,2,...,K - 1]
    P_fast = Lambda(lambda x: x[:, ::2, ...])(P)  # S_0, S_2, S_5 ,... is P 1D?
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l / 2), -1))(P_diff_fast)  # M_fast_[1,2,...,K/2 - 1]
    return P_diff_slow, P_diff_fast


def c1D(x, filters, kernel):
    x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def block(x, filters):
    x = c1D(x, filters, 3)
    x = c1D(x, filters, 3)
    return x


def d1D(x, filters):
    x = Dense(filters, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def build_FM(frame_l=32, joint_n=22, joint_d=2, feat_d=231, filters=16):
    M = Input(shape=(frame_l, feat_d))
    P = Input(shape=(frame_l, joint_n, joint_d))

    diff_slow, diff_fast = pose_motion(P, frame_l)

    # Modeling Joint Correlations by an Embedding
    x = c1D(M, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow, filters * 2, 1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

    x_d_fast = c1D(diff_fast, filters * 2, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 3)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

    x = concatenate([x, x_d_slow, x_d_fast])
    x = block(x, filters * 2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 8)
    x = SpatialDropout1D(0.1)(x)

    return Model(inputs=[M, P], outputs=x)


def build_DD_Net(frame_l=32, joint_n=22, joint_d=3, feat_d=231, clc_num=14, filters=16):
    M = Input(name='M', shape=(frame_l, feat_d))
    P = Input(name='P', shape=(frame_l, joint_n, joint_d))

    FM = build_FM(frame_l, joint_n, joint_d, feat_d, filters)

    x = FM([M, P])

    x = GlobalMaxPool1D()(x)

    x = d1D(x, 128)
    x = Dropout(0.5)(x)
    x = d1D(x, 128)
    x = Dropout(0.3)(x)
    x = Dense(clc_num, activation='softmax')(x)

    ######################Self-supervised part
    model = Model(inputs=[M, P], outputs=x)
    return model
