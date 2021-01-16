from preprocess import preprocess
from keras.models import load_model
import numpy as np
import cv2
from optic_flow import optic_flow
from imread import imread
import sys

size = int(sys.argv[1])
h, w = 66, 200
sample = np.zeros((size, h, w, 3))
speed = np.loadtxt('train.txt')[:size]

for i in range(size):
    curr = imread(i)
    next = imread(i+1)

    curr = preprocess(curr)
    next = preprocess(next)

    diff = optic_flow(curr, next)
    sample[i] = diff

model = load_model('model')

p = model(sample/256 - 0.5)
p = p.numpy().reshape(size);
print(speed[:10], '\n')
print(p[:10], '\n')

mse = np.mean((speed-p)**2)
print(mse)
