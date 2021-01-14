from model import model as make_model
from optic_flow import optic_flow
from change_brightness import change_brightness
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import math
from sklearn.utils import shuffle

X_label = np.loadtxt('train.txt')
X_frames = 20400

batch_size = 64
batch = X_frames // batch_size

h, w, _ = cv2.imread('frame_train/0.jpg').shape

def generator_train():
    input = np.zeros((batch_size, h, w, 3), dtype=np.float32)
    output = np.zeros((batch_size))

    n = 0
    while True:
        j = 0
        for i in range(batch_size * n, batch_size * n + batch_size):
            img1 = cv2.imread('frame_train/%d.jpg' % i)
            img2 = cv2.imread('frame_train/%d.jpg' % (i+1))
            curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            bright_factor = 0.2 + np.random.uniform()
            curr = change_brightness(curr, bright_factor)
            next = change_brightness(next, bright_factor)

            diff = optic_flow(curr, next)
            s1 = X_label[i]
            s2 = X_label[i+1]

        input[j] = diff
        output[j] = np.mean([s1, s2])
        j += 1
        input, output = shuffle(input, output)
        yield (input/256-.5, output)
        n += 1
        if n == batch:
            n = 0

import sys
adam = Adam(float(sys.argv[1]),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08)

es = EarlyStopping(monitor='loss',
                   min_delta=1e-4,
                   patience=1)

model = make_model((h, w, 3))

model.compile(optimizer=adam,
              loss='mse')

model.fit(generator_train(),
          batch_size=batch_size,
          epochs=100,
          steps_per_epoch=X_frames//batch_size,
          callbacks=[es],
          verbose=1)

model.save('model')

model.summary()
