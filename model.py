import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, ELU, Dropout, Flatten, Dense
from keras.optimizers import Adam

def model():
    m = Sequential()

    m.add(keras.Input(shape=(100, 200, 3)))

    m.add(Conv2D(24,
                     (5,5),
                     strides=(2,2),
                     padding = 'valid',
                     kernel_initializer = 'he_normal',
                     name = 'conv1'))
    m.add(ELU())
    m.add(Conv2D(36,
                     (5,5),
                     strides=(2,2),
                     padding = 'valid',
                     kernel_initializer = 'he_normal',
                     name = 'conv2'))
    m.add(ELU())
    m.add(Conv2D(48,
                     (5,5),
                     strides=(2,2),
                     padding = 'valid',
                     kernel_initializer = 'he_normal',
                     name = 'conv3'))
    m.add(ELU())
    m.add(Dropout(0.5))
    m.add(Conv2D(64,
                     (3,3),
                     strides = (1,1),
                     padding = 'valid',
                     kernel_initializer = 'he_normal',
                     name = 'conv4'))
    m.add(ELU())
    m.add(Conv2D(64,
                     (3,3),
                     strides= (1,1),
                     padding = 'valid',
                     kernel_initializer = 'he_normal',
                     name = 'conv5'))

    m.add(Flatten(name = 'flatten'))
    m.add(ELU())

    m.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    m.add(ELU())
    m.add(Dense(50,  kernel_initializer = 'he_normal', name = 'fc2'))
    m.add(ELU())
    m.add(Dense(10,  kernel_initializer = 'he_normal', name = 'fc3'))
    m.add(ELU())
    m.add(Dense(1,   kernel_initializer = 'he_normal', name='output'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    m.compile(optimizer = adam, loss = 'mse')

    return m

