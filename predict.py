from keras.models import load_model

from imread import imread
from optic_flow import optic_flow
from progress import progress

import numpy as np
import sys

from globl import h, w

size = int(sys.argv[1])

sample = np.zeros((1, h, w, 3))
speed = np.loadtxt('train.txt')[19000: 19000 + size]

index = np.arange(19000, 19000 + size)

model = load_model('model.h5')
predict = np.zeros((size))

for i in range(len(index)):
    j = index[i]

    curr = imread(j)
    next = imread(j + 1)

    diff = optic_flow(curr, next)

    sample[0] = diff
    predict[i] = model(sample).numpy()[0]

    progress(i + 1, size)

mse = np.mean((speed - predict) ** 2)

print(speed[:20], '\n')
print(predict[:20], '\n')
print(mse)
