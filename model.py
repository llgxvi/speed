from keras import Sequential
from keras.layers import Lambda, Conv2D
from keras.layers import Dropout, Flatten, Dense

from globl import h, w

def model():
    m = Sequential()

    m.add(Lambda(lambda x: x / 255, input_shape=(h, w, 3)))

    m.add(Conv2D(16, (5,5), 2,
                 activation='elu',
                 kernel_initializer='he_normal'))

    m.add(Conv2D(32, (5,5), 2,
                 activation='elu',
                 kernel_initializer='he_normal'))

    m.add(Conv2D(32, (3,3), 2,
                 activation='elu',
                 kernel_initializer='he_normal'))

    m.add(Dropout(0.5))

    m.add(Conv2D(64, (3,3), 1,
                 activation='elu',
                 kernel_initializer='he_normal'))

    m.add(Conv2D(64, (3,3), 1,
                  activation='elu',
                  kernel_initializer='he_normal'))

    m.add(Flatten())

    m.add(Dense(100,
                activation='elu',
                kernel_initializer='he_normal'))

    m.add(Dropout(0.2))

    m.add(Dense(10,
                activation='elu',
                kernel_initializer='he_normal'))

    m.add(Dense(1, kernel_initializer='he_normal'))

    return m
