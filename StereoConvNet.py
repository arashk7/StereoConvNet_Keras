import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, ZeroPadding2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import _reduced_kernel_size_for_small_input
from keras import losses, activations, optimizers, initializers


def get_model():
    Input_1 = Input(shape=(100, 150, 2), name='Input_1')

    Convolution2D_1 = Convolution2D(name='Convolution2D_1', nb_filter=16, kernel_size=(7, 7), activation='relu',
                                    border_mode='same')(Input_1)

    BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(Convolution2D_1)

    MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1')(BatchNormalization_1)

    Convolution2D_3 = Convolution2D(name='Convolution2D_3', nb_filter=32, nb_row=5, activation='relu', nb_col=5,
                                    border_mode='same')( MaxPooling2D_1)

    BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2')(Convolution2D_3)

    MaxPooling2D_2 = MaxPooling2D(name='MaxPooling2D_2')(BatchNormalization_2)

    Convolution2D_4 = Convolution2D(name='Convolution2D_4', nb_filter=64, nb_row=3, activation='relu', nb_col=3,
                                    border_mode='same')( MaxPooling2D_2)

    BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3')(Convolution2D_4)

    MaxPooling2D_3 = MaxPooling2D(name='MaxPooling2D_3')(BatchNormalization_3)

    Convolution2D_5 = Convolution2D(name='Convolution2D_5', nb_filter=128, nb_row=3, activation='relu', nb_col=3,
                                    border_mode='same')( MaxPooling2D_3)

    BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4')(Convolution2D_5)

    Convolution2D_6 = Convolution2D(name='Convolution2D_6', nb_filter=32, nb_row=3, border_mode='same',
                                    activation='relu', nb_col=3)(BatchNormalization_4)

    BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5')(Convolution2D_6)

    UpSampling2D_1 = UpSampling2D(name='UpSampling2D_1')(BatchNormalization_5)

    Convolution2D_7 = Convolution2D(name='Convolution2D_7', nb_filter=16, nb_row=3, border_mode='same',
                                    activation='relu', nb_col=3)(UpSampling2D_1)

    BatchNormalization_6 = BatchNormalization(name='BatchNormalization_6')(Convolution2D_7)

    UpSampling2D_2 = UpSampling2D(name='UpSampling2D_2')(BatchNormalization_6)

    Convolution2D_8 = Convolution2D(name='Convolution2D_8', nb_filter=8, nb_row=5, border_mode='same',
                                    activation='relu', nb_col=5)(UpSampling2D_2)

    BatchNormalization_7 = BatchNormalization(name='BatchNormalization_7')(Convolution2D_8)

    UpSampling2D_3 = UpSampling2D(name='UpSampling2D_3')(BatchNormalization_7)

    ZeroPadding2D_1 = ZeroPadding2D(name='ZeroPadding2D_1', padding=(2, 3))(UpSampling2D_3)

    Convolution2D_9 = Convolution2D(name='Convolution2D_9', nb_filter=1, nb_row=7, border_mode='same',
                                    activation='relu', nb_col=7)(ZeroPadding2D_1)

    BatchNormalization_8 = BatchNormalization(name='BatchNormalization_8')(Convolution2D_9)

    model = Model([Input_1], [BatchNormalization_8])
    return model


from keras.optimizers import *


def get_optimizer():
    return SGD()


def is_custom_loss_function():
    return False


def get_loss_function():
    return 'mean_squared_error'


def get_batch_size():
    return 1


def get_num_epoch():
    return 100


