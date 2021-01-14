from keras.models import load_model
import numpy as np
import cv2
from optic_flow import optic_flow
from change_brightness import change_brightness

h, w, _ = cv2.imread('frame_train/0.jpg').shape
sample = np.zeros((10, h, w, 3))
speed = np.loadtxt('train.txt')[:10]

for i in range(10):
    img1 = cv2.imread('frame_train/%d.jpg' % i)
    img2 = cv2.imread('frame_train/%d.jpg' % (i+1))
    curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    bright_factor = 0.2 + np.random.uniform()
    curr = change_brightness(curr, bright_factor)
    next = change_brightness(next, bright_factor)

    diff = optic_flow(curr, next)
    sample[i] = diff

model = load_model('model')
p = model(sample)
p = p.numpy().reshape(10)
print(speed, '\n')
print(p, '\n\n')

mse = np.mean((speed-p)**2)
print(mse)
