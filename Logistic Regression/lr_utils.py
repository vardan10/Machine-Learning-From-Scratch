import numpy as np
from scipy import ndimage
import scipy
import os





def load_dataset():

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

    train_set_y_orig = np.asarray(train_set_y_orig, dtype=np.int).reshape(m,1).T
    print (train_set_y_orig.shape)

    classes = ['non-cat','cat']
    classes = np.array(classes)

    return train_set_x_orig, train_set_y_orig, train_set_x_orig, train_set_y_orig, classes
