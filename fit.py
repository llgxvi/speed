from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

from imread import imread
from preprocess import preprocess
from optic_flow import optic_flow
from model import model as make_model

import numpy as np
import cv2
import sys

h, w = 66, 200

X_label = np.loadtxt('train.txt')
X_size = X_label.shape[0]
batch_size = 64
v_size = 100
lr = 1e-4
epoch = 100

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
index = np.arange(X_size - 1)

def generator_x():
    x = np.zeros((batch_size, h, w, 3))
    y = np.zeros((batch_size))

    while True:
        mini = np.random.choice(index, batch_size)

        for i in range(len(mini)):
            bf = 0.2 + np.random.uniform()

            j = mini[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = preprocess(curr, bf)
            next = preprocess(next, bf)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j:j+1])

        yield (x / 256 - 0.5, y)

# vx: validation batch
def generator_vx():
    x = np.zeros((v_size, h, w, 3))
    y = np.zeros((v_size))

    a = np.random.choice(index, v_size)

    while True:
        for i in range(len(a)):
            j = a[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = preprocess(curr)
            next = preprocess(next)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j:j+1])

        yield (x / 256 - 0.5, y)

adam = Adam(lr, epsilon=1e-07)

es = EarlyStopping(monitor='val_loss', min_delta=0.01)

model = make_model((h, w, 3))
model = load_model('model')

model.compile(optimizer=adam, loss='mse')

model.fit(generator_x(),
          batch_size=batch_size,
          epochs=epoch,
          steps_per_epoch=X_size // batch_size,
          validation_data=generator_vx(),
          validation_steps=1,
          callbacks=[es],
          verbose=1)

model.save('model')

model.summary()
