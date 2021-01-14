from model import model as make_model
from optic_flow import optic_flow
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.models import load_model

speed = np.loadtxt('train.txt')
frame_train = 20400
frame_test = 10798

frame_train = 1000
frame_test = 100

h, w, _ = cv2.imread('frame_train/0.jpg').shape
input = np.zeros((frame_train - 1, h, w, 3), dtype='float16')
output = np.zeros((frame_train - 1))

input_test = np.zeros((frame_test - 1, h, w, 3), dtype='float16')

for i in range(frame - 1):
    img1 = cv2.imread('frame_train/%d.jpg' % i)
    img2 = cv2.imread('frame_train/%d.jpg' % (i+1))
    curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    diff = optic_flow(curr, next)
  
    input[i] = diff
    output[i] = np.mean([speed[i], speed[i+1]])

adam = Adam(0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-01)

model = make_model()

model.compile(optimizer=adam, 
              loss='mse')

model.fit(x=input,
          y=output,
          batch_size=2,
          epochs=100,
          verbose=2)

model.save('model')

model.summary()

for i in range(frame_test - 1):
    img1 = cv2.imread('frame_test/%d.jpg' % i)
    img2 = cv2.imread('frame_test/%d.jpg' % (i+1))
    curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    diff = optic_flow(curr, next)
  
    input_test[i] = diff

m2 = load_model('model')
m2.predict(input_test)
