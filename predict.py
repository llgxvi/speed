from keras.models import load_model
import numpy as np
import cv2
from optic_flow import optic_flow
from change_brightness import change_brightness
from imread import imread

h, w, _ = imread(0).shape
sample = np.zeros((10, h, w, 3))
speed = np.loadtxt('train.txt')[:10]
bright_factor = 0.2 + np.random.uniform()

for i in range(10):
    curr = imread(i)
    next = imread(i+1)

    #curr = change_brightness(curr, bright_factor)
    #next = change_brightness(next, bright_factor)

    diff = optic_flow(curr, next)
    sample[i] = diff/256 - .5

model = load_model('model')

p = model(sample)
p = p.numpy().reshape(10);
print(speed, '\n')
print(p, '\n')

mse = np.mean((speed-p)**2)
print(mse)
