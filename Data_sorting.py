import numpy as np
import matplotlib.pyplot as plot
import sys
import os
import pickle
import pdb
import scipy
from PIL import Image
from sklearn.model_selection import train_test_split


x_train_path = "data\\task2\\task2_X_train.npy"
y_train_path = "data\\task2\\task2_y_train.npy"
x_test_path = "data\\task2\\task2_X_test.npy"
y_test_path = "data\\task2\\task2_y_test.npy"


def recenter_pixels(x):
    """Recenter pixel values to be between 0 and 255 to recreate images. Data could only
    be downloaded as numpy array, so this is necessary to transform back into images for viewing"""
    maximum = np.amin(x, axis=(0, 1, 2))
    minimum = np.amax(x, axis=(0, 1, 2))
    modification = (maximum + minimum) / 2
    for i, pixel_mod in enumerate(modification):
        x[:, :, :, i] -= pixel_mod
        x[:, :, :, i] += 255/2
    return x


y_train = np.load(y_train_path, allow_pickle=True)
x_train = recenter_pixels(np.load(x_train_path, allow_pickle=True))
x_test = recenter_pixels(np.load(x_test_path, allow_pickle=True))
y_test = np.load(y_test_path, allow_pickle=True)
pdb.set_trace()
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.07, random_state=23)
ni, nw, nh, nc = x_train.shape
# Number of categories in classification dataset
ncat = len(y_train[1])


def reverse_onehot(y, ncat):
    """Reverse the on-hot encoding of the labels. Changes labels to integer value starting at 0 for the
    first category, 1 for the second, etc. """
    y_new = np.zeros((y.shape[0], 1))
    for i in range(ncat):
        cat_i = np.zeros(ncat)
        cat_i[i] += 1
        indices = list(set(np.where(y == cat_i)[0]))
        y_new[indices] = i
    y_new = y_new.reshape(y_new.shape[0])
    return y_new


# Reshape vectors to reverse one-hot encoding
y_train = reverse_onehot(y_train, ncat)
y_validation = reverse_onehot(y_validation, ncat)
y_test = reverse_onehot(y_test, ncat)

images_folder = os.getcwd()+"\data\Images\Task2"
sort_key = {float(0): "Damaged", float(1): "Undamaged"}

# Sort test images
# print("SORTING TEST IMAGES:\n")
# for i, image in enumerate(x_test):
#     category = sort_key[y_test[i]]
#     im = Image.fromarray(np.uint8(image))
#     im.save(images_folder+"\Test"+os.path.sep+category+os.path.sep+"img"+str(i)+".png")
#     if i % 100 == 0:
#         print("{} out of {} test images saved".format(i, len(x_test)))
# print("All test images saved\n")

# Sort validation images
print("SORTING TRAIN IMAGES:\n")
for i, image in enumerate(x_validation):
    category = sort_key[y_validation[i]]
    im = Image.fromarray(np.uint8(image))
    im.save(images_folder+"\Validation"+os.path.sep+category+os.path.sep+"img"+str(i)+".png")
    if i % 100 == 0:
        print("{} out of {} validation images saved".format(i, len(x_validation)))
print("All validation images saved\n")


# Sort training images
print("SORTING TRAIN IMAGES:\n")
for i, image in enumerate(x_train):
    category = sort_key[y_train[i]]
    im = Image.fromarray(np.uint8(image))
    im.save(images_folder+"\Train"+os.path.sep+category+os.path.sep+"img"+str(i)+".png")
    if i % 100 == 0:
        print("{} out of {} train images saved".format(i, len(x_train)))
