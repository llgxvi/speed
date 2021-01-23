import cv2 as cv
h, w, _ = cv.imread('train_frames/0.jpg').shape

import numpy as np
label = np.loadtxt('train.txt')
