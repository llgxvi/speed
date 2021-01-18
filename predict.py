from keras.models import load_model

from imread import imread
from preprocess import preprocess
from optic_flow import optic_flow
from progress import progress

import numpy as np
import sys

h, w = 66, 200

size = int(sys.argv[1])

sample = np.zeros((1, h, w, 3))
speed = np.loadtxt('train.txt')[19000:19000+size]

index = np.arange(19000, 19000+size)

model = load_model('model')
predict = np.zeros((size))

for j in range(len(index)):
    i = index[j]
    curr = imread(i)
    next = imread(i + 1)

    curr = preprocess(curr)
    next = preprocess(next)

    diff = optic_flow(curr, next)

    sample[0] = diff
    predict[j] = model(sample / 256 - 0.5).numpy()[0]

    progress(j+1, size)

mse = np.mean((speed - predict) ** 2)

print(speed[:20], '\n')
print(predict[:20], '\n')
print(mse)
