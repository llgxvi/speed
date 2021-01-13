from model import model as make_model
from optic_flow import optic_flow
import numpy as np
import cv2

speed = np.loadtxt("train.txt")
frame = 20400
batch = 10

input = np.zeros((100,100,200,3))
output = np.zeros((100))

n = 0
j = 0
while(n < batch * 2):
    img1 = cv2.imread("stretched/%d.jpg" % n)
    img2 = cv2.imread("stretched/%d.jpg" % (n+1))
    prev = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    diff = optic_flow(prev, next)
    input[j] = diff
    output[j] = np.mean([speed[n], speed[n+1]])
    n += 2

print('Shape of input, output')
print(input.shape, output.shape)

model = make_model()

model.fit(x=input,
          y=output,
          batch_size=None,
          epochs=1,
          verbose=2)
