import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import os


def load_dataset2():

    train_set_x_orig = []
    train_set_y_orig = []
    for xyz in os.listdir('real_dataset/positive'):
        fname = 'real_dataset/positive/' + xyz
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
        train_set_x_orig.append(my_image);
        train_set_y_orig.append(1);
    for xyz in os.listdir('real_dataset/negative'):
        fname = 'real_dataset/negative/' + xyz
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
        train_set_x_orig.append(my_image);
        train_set_y_orig.append(0);

    m = len(os.listdir('real_dataset/positive'))+len(os.listdir('real_dataset/negative'))

    train_set_x_orig = np.array(train_set_x_orig).reshape(m, 64 * 64 * 3).T
    print(train_set_x_orig.shape)

    #train_set_y_orig = np.array(train_set_x_orig).reshape(1,len(os.listdir('real_dataset/positive'))+len(os.listdir('real_dataset/negative'))).T

    train_set_y_orig = np.asarray(train_set_y_orig, dtype=np.int).reshape(m,1).T
    print (train_set_y_orig.shape)

    classes = ['non-cat','cat']
    classes = np.array(classes)

    return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig, classes



#load_dataset2()




#(1, 209)