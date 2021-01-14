from model import model as make_model
from optic_flow import optic_flow
import numpy as np
import cv2
from keras.optimizers import Adam
import math

speed = np.loadtxt('train.txt')
train_frames = 20400
test_frames = 10798

batch_size = 32
batch = math.floor(train_frames / batch_size)

h, w, _ = cv2.imread('frame_train/0.jpg').shape

def generator_train():
    input = np.zeros((batch_size, h, w, 3), dtype='float16')
    output = np.zeros((batch_size))

    n = 0
    while True and n < batch:
        j = 0
        for i in range(batch_size * n, batch_size * n + batch_size):
            img1 = cv2.imread('frame_train/%d.jpg' % i)
            img2 = cv2.imread('frame_train/%d.jpg' % (i+1))
            curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            diff = optic_flow(curr, next)

        input[j] = diff
        output[j] = np.mean([speed[j], speed[j+1]])
        j += 1
        yield (input/256-.05, output)
        n += 1

import sys
adam = Adam(float(sys.argv[1]),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08)

model = make_model((h, w, 3))

model.compile(optimizer=adam,
              loss='mse')

model.fit(generator_train(),
          batch_size=batch_size,
          epochs=10,
          verbose=1)

model.save('model')

model.summary()
