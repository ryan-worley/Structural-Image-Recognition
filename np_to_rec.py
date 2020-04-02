import numpy as np
import matplotlib.pyplot as plot
import sys
import os
import pickle
import pdb
import scipy
from sklearn.model_selection import train_test_split
import mxnet as mx


def reverse_onehot(y, ncat):
    y_new = np.zeros((y.shape[0], 1))
    for i in range(ncat):
        cat_i = np.zeros(ncat)
        cat_i[i] += 1
        indices = list(set(np.where(y == cat_i)[0]))
        y_new[indices] = float(i)
    y_new = y_new.reshape(y_new.shape[0])
    return y_new


def image_encode(n, img, label):
    img = np.ndarray.astype(img, "int16")
    img = np.ndarray.tobytes(img, order=None)
    header = mx.recordio.IRHeader(0, float(label), int(n), 0)
    s = mx.recordio.pack(header, img)
    return s

def make_recordio(x, y, _type):
    fname = "data_rec\\task2_rec" + _type
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(fname_idx, fname_rec, 'w')
    for i, img in enumerate(x):
        encoded_image = image_encode(i, img, y[i])
        record.write_idx(i, encoded_image)
    record.close()
    return


x_train_path = "data\\task2_X_train.npy"
y_train_path = "data\\task2_y_train.npy"
x_test_path = "data\\task2_X_test.npy"
y_test_path = "data\\task2_y_test.npy"

y_train = np.load(y_train_path, allow_pickle=True)
x_train = np.load(x_train_path, allow_pickle=True)
y_test = np.load(y_test_path, allow_pickle=True)
x_test = np.load(x_test_path, allow_pickle=True)

pdb.set_trace()

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.07, random_state=23)

# Reverse 1 hot encode the validation
y_train = reverse_onehot(y_train, 2)
y_test = reverse_onehot(y_test, 2)
y_validation = reverse_onehot(y_validation, 2)

# Create recordio files
make_recordio(x_train, y_train, "train")
make_recordio(x_test, y_test, "test")
make_recordio(x_validation, y_validation, "val")


pdb.set_trace()



