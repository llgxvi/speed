from model import model as make_model
from optic_flow import optic_flow
import numpy as np
import cv2
from keras.optimizers import Adam

speed = np.loadtxt("train.txt")
frame = 20400
batch = 10

input = np.zeros((10,100,200,3))
output = np.zeros((10))

n = 0
j = 0
while(n < 19):
    img1 = cv2.imread("stretched/%d.jpg" % n)
    img2 = cv2.imread("stretched/%d.jpg" % (n+1))
    curr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    diff = optic_flow(curr, next)
    input[j] = diff
    output[j] = np.mean([speed[n], speed[n+1]])
    n += 2

print('Shape of input, output')
print(input.shape, output.shape)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = make_model()

model.compile(optimizer=adam, 
              loss='mse')

model.fit(x=input,
          y=output,
          batch_size=4,
          epochs=1,
          verbose=2)
