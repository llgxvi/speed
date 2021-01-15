from model import model as make_model
from preprocess import preprocess
from optic_flow import optic_flow
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.utils import shuffle
from imread import imread
import sys

X_label = np.loadtxt('train.txt')
X_size = X_label.shape[0]
batch_size = 64
lr = 1e-4

l_ = len(sys.argv)
if l_ > 1:
    lr = float(sys.argv[1])
if l_ > 2:
    X_size = int(sys.argv[2])
if l_ > 3:
    batch_size = int(sys.argv[3])

batch = X_size // batch_size

h, w = 66, 200

def generator_x():
    x = np.zeros((batch_size, h, w, 3))
    y = np.zeros((batch_size))

    b = 0 # batch index
    while True:
        i = 0 # img index inside batch
        for j in range(batch_size * b, batch_size * (b + 1)):
            bf = 0.2 + np.random.uniform()

            curr = imread(j)
            next = imread(j + 1) # index out of range is ok in this case

            curr = preprocess(curr, bf)
            next = preprocess(next, bf)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j:j+1])
            i += 1

        x, y = shuffle(x, y)
        yield (x/256 - 0.5, y)

        b += 1
        if b == batch:
            b = 0

# vx: validation batch
def generator_vx():
    x = np.zeros((10, h, w, 3))
    y = np.zeros((10))

    while True:
        from random import randint
        r = randint(0, X_size - 11)

        i = 0 # img index inside batch
        for j in range(r, r + 10):
            curr = imread(j)
            next = imread(j+1)

            curr = preprocess(curr)
            next = preprocess(next)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j:j+1])
            i += 1

        x, y = shuffle(x, y)
        yield (x/256 - 0.5, y)

adam = Adam(lr, epsilon=1e-07)

es = EarlyStopping(monitor='loss', patience=10)

model = make_model((h, w, 3))
model = load_model('model')

model.compile(optimizer=adam, loss='mse')

model.fit(generator_x(),
          batch_size=batch_size,
          epochs=100,
          steps_per_epoch=X_size // batch_size,
          validation_data=generator_vx(),
          validation_steps=1,
          callbacks=[es],
          verbose=1)

model.save('model')

model.summary()
