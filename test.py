import cv2
import numpy as np

from preprocess import preprocess
from optic_flow import optic_flow
from progress import progress

from keras.models import load_model

cap = cv2.VideoCapture('test.mp4')

frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = load_model('model')

speed = np.zeros((frameC))

tmp = None
for i in range(frameC):
    ret, img = cap.read()

    if ret is False:
        print('Failure at frame', i)
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if tmp is None:
        tmp = img
        continue

    curr = preprocess(tmp)
    next = preprocess(img)

    tmp = img

    diff = optic_flow(curr, next)

    sample = np.zeros((1, 66, 200, 3))
    sample[0] = diff / 256 - 0.5
    predict = model(sample)

    speed[i-1] = predict.numpy()[0]

    progress(i, frameC)

cap.release()

with open('test.txt', 'w') as f:
    f.write('\n'.join(map(str, speed)))
