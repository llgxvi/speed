from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

from imread import imread
from preprocess import preprocess
from optic_flow import optic_flow
from model import model as make_model

import numpy as np
import sys

h, w = 66, 200

X_label = np.loadtxt('train.txt')
X_size = 16000
V_size = 4400 - 2
batch_size = 32
v_size = 100
lr = 1e-4
epoch = 10

if False:
    L = len(sys.argv)
    if L > 1:
        lr = float(sys.argv[1])
    if L > 2:
        X_size = int(sys.argv[2])
    if L > 3:
        batch_size = int(sys.argv[3])
    if L > 4:
        epoch = int(sys.argv[4])

batch = X_size // batch_size

def generator_x():
    x = np.zeros((batch_size, h, w, 3))
    y = np.zeros((batch_size))

    c = 0
    while True:
        if c == 0:
            index = np.arange(X_size)
            index.reshape(-1, 2)
            np.random.shuffle(index)
            index.ravel()

        mini = index[batch_size * c: batch_size * (c + 1)]

        for i in range(len(mini)):
            bf = 0.2 + np.random.uniform()

            j = mini[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = preprocess(curr, bf)
            next = preprocess(next, bf)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j: j+1])

        yield (x / 256 - 0.5, y)

        if c == batch - 1:
            c = 0
        else:
            c += 1

# vx: validation batch
def generator_vx():
    x = np.zeros((v_size, h, w, 3))
    y = np.zeros((v_size))

    index = np.arange(X_size, V_size)

    c = 0
    while True:
        slice = index[v_size * c: v_size * (c + 1)]
        c += 1

        for i in range(len(slice)):
            j = slice[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = preprocess(curr)
            next = preprocess(next)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j: j+1])

        yield (x / 256 - 0.5, y)

adam = Adam(lr, epsilon=1e-07)

es = EarlyStopping(monitor='val_loss',
                   min_delta=0.001,
                   patience=3)

# model = make_model((h, w, 3))
model = load_model('model')

model.compile(optimizer=adam, loss='mse')

history = model.fit(generator_x(),
          epochs=epoch,
          batch_size=batch_size,
          steps_per_epoch=X_size // batch_size,
          validation_data=generator_vx(),
          validation_steps=V_size // v_size,
          callbacks=[es],
          verbose=1)

model.save('model')

import json
with open('history.json', 'w') as f:
    json.dump(history.history, f)

model.summary()
