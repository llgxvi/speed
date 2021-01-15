from model import model as make_model
from optic_flow import optic_flow
from change_brightness import change_brightness
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from imread import imread
import sys

X_label = np.loadtxt('train.txt')
X_size = X_label.shape[0]
batch_size = 64
lr = 0.001

l_ = len(sys.argv)
if l_ > 1:
    lr = float(sys.argv[1])
if l_ > 2:
    X_size = int(sys.argv[2])
if l_ > 3:
    batch_size = int(sys.argv[3])

batch = X_size // batch_size

h, w, _ = imread(0).shape

def generator_x():
    x = np.zeros((batch_size, h, w, 3))
    y = np.zeros((batch_size))

    b = 0 # batch index
    while True:
        bright_factor = 0.2 + np.random.uniform()

        i = 0
        for j in range(batch_size * b, batch_size * (b + 1)):
            curr = imread(j)
            next = imread(j + 1) # index out of range is ok in this case

            curr = change_brightness(curr, bright_factor)
            next = change_brightness(next, bright_factor)

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
        bright_factor = 0.2 + np.random.uniform()

        from random import randint
        r = randint(0, X_size - 11)

        i = 0
        for j in range(r, r + 10):
            curr = imread(j)
            next = imread(j+1)

            curr = change_brightness(curr, bright_factor)
            next = change_brightness(next, bright_factor)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(X_label[j:j+1])
            i += 1

        x, y = shuffle(x, y)
        yield (x/256 - 0.5, y)

adam = Adam(lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08)

es = EarlyStopping(monitor='loss',
                   min_delta=1e-4,
                   patience=2)

model = make_model((h, w, 3))

model.compile(optimizer=adam,
              loss='mse')

model.fit(generator_x(),
          batch_size=batch_size,
          epochs=100,
          steps_per_epoch=X_size // batch_size,
          validation_data=generator_vx(),
          validation_steps=2,
          callbacks=[es],
          verbose=1)

model.save('model')

model.summary()
