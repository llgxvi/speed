from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

from imread import imread
from change_brightness import change_brightness
from optic_flow import optic_flow
from model import model as make_model

import numpy as np
import sys

from globl import h, w

X_label = np.loadtxt('train.txt')
X_size = 16000
V_size = 3000
batch_size = 16
v_size = 100
lr = 1e-4
epoch = 5

batch = X_size // batch_size
batch_v = V_size // v_size

index_x = np.arange(X_size)
index_v = np.arange(X_size, X_size + V_size)

def generator_x():
    x = np.zeros((batch_size, h, w, 3))
    y = np.zeros((batch_size))

    c = 0
    while True:
        mini = index_x[batch_size * c: batch_size * (c + 1)]

        for i in range(len(mini)):
            bf = np.random.uniform(0.2, 1)

            j = mini[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = change_brightness(curr, bf)
            next = change_brightness(next, bf)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j: j + 2])

        yield (x / 256 - 0.5, y)

        if c == batch - 1:
            c = 0
        else:
            c += 1

def generator_v():
    x = np.zeros((v_size, h, w, 3))
    y = np.zeros((v_size))

    c = 0
    while True:
        progress(c + 1, V_size)

        slice = index_v[v_size * c: v_size * (c + 1)]

        for i in range(len(slice)):
            j = slice[i]

            curr = imread(j)
            next = imread(j + 1)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j: j + 2])

        yield (x / 256 - 0.5, y)

        if c == batch_v - 1:
            c = 0
        else:
            c += 1

adam = Adam(lr, epsilon=1e-07)

es = EarlyStopping(monitor='val_loss',
                   min_delta=0.001,
                   patience=100)

model = make_model()
# model = load_model('model')

model.compile(optimizer=adam, loss='mse')

history = model.fit(generator_x(),
          epochs=epoch,
          batch_size=batch_size,
          steps_per_epoch=batch,
          validation_data=generator_v(),
          validation_steps=batch_v,
          callbacks=[es],
          verbose=1)

model.save('model')

print(history.history['loss'])
print(history.history['val_loss'])

model.summary()
