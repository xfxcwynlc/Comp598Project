#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
from model import *
import keras


DD_Net = build_DD_Net(C.frame_l, C.joint_n, C.joint_d, C.feat_d, C.clc_coarse, C.filters)
DD_Net.summary()

Train = pickle.load(open("SHREC/train.pkl", "rb"))
Test = pickle.load(open("SHREC/test.pkl", "rb"))

X_0 = []
X_1 = []
Y = []
for i in tqdm(range(len(Train['pose']))):
    p = np.copy(Train['pose'][i]).reshape([-1, 22, 3])
    p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_coarse)
    label[Train['coarse_label'][i] - 1] = 1

    M = get_CG(p, C)

    X_0.append(M)
    X_1.append(p)
    Y.append(label)

X_0 = np.stack(X_0)
X_1 = np.stack(X_1)
Y = np.stack(Y)

X_test_0 = []
X_test_1 = []
Y_test = []
for i in tqdm(range(len(Test['pose']))):
    p = np.copy(Test['pose'][i]).reshape([-1, 22, 3])
    p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_coarse)
    label[Test['coarse_label'][i] - 1] = 1

    M = get_CG(p, C)

    X_test_0.append(M)
    X_test_1.append(p)
    Y_test.append(label)

X_test_0 = np.stack(X_test_0)
X_test_1 = np.stack(X_test_1)
Y_test = np.stack(Y_test)

lr = 1e-3
DD_Net.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr), metrics=['accuracy'])
lrScheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=5e-6)

history = DD_Net.fit([X_0, X_1], Y,
                     batch_size=len(Y),
                     epochs=600,
                     verbose=True,
                     shuffle=True,
                     callbacks=[lrScheduler],
                     validation_data=([X_test_0, X_test_1], Y_test)
                     )

DD_Net.save("DD_Net.h5")
