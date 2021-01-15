from keras import Sequential
from keras.layers import Input, Conv2D
from keras.layers import ELU, Dropout, Flatten, Dense

def model(input_shape):
    m = Sequential()

    m.add(Input(shape=input_shape))

    m.add(Conv2D(24, (5,5),
                 strides=(2,2),
                 padding='valid',
                 kernel_initializer='he_normal'))
    m.add(ELU())

    m.add(Conv2D(36, (5,5),
                 strides=(2,2),
                 padding='valid',
                 kernel_initializer='he_normal'))
    m.add(ELU())

    m.add(Conv2D(48, (5,5),
                 strides=(2,2),
                 padding='valid',
                 kernel_initializer='he_normal'))
    m.add(ELU())

    m.add(Dropout(0.5))

    m.add(Conv2D(64, (3,3),
                 strides=(1,1),
                 padding='valid',
                 kernel_initializer='he_normal'))
    m.add(ELU())

    m.add(Conv2D(64, (3,3),
                 strides=(1,1),
                 padding='valid',
                 kernel_initializer='he_normal'))

    m.add(Flatten())
    m.add(ELU())

    m.add(Dense(100, kernel_initializer='he_normal'))
    m.add(ELU())
    m.add(Dense(50,  kernel_initializer='he_normal'))
    m.add(ELU())
    m.add(Dense(10,  kernel_initializer='he_normal'))
    m.add(ELU())

    m.add(Dense(1,   kernel_initializer='he_normal'))

    return m

