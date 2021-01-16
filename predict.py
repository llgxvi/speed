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

predict = model(sample / 256 - 0.5)
predict = predict.numpy().reshape(size)
mse = np.mean((speed - predict) ** 2)

print(speed[:10], '\n')
print(predict[:10], '\n')
print(mse)
