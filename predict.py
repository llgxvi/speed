from keras.models import load_model

from imread import imread
from preprocess import preprocess
from optic_flow import optic_flow

import numpy as np
import sys

h, w = 66, 200

size = int(sys.argv[1])

sample = np.zeros((size, h, w, 3))
speed = np.loadtxt('train.txt')[:size]

for i in range(size):
    curr = imread(i)
    next = imread(i + 1)

    curr = preprocess(curr)
    next = preprocess(next)

    diff = optic_flow(curr, next)
    sample[i] = diff

model = load_model('model')

p = model(sample / 256 - 0.5)
p = p.numpy().reshape(size);
print(speed[:10], '\n')
print(p[:10], '\n')

mse = np.mean((speed-p)**2)
print(mse)
