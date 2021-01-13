from model import model as make_model
from optic_flow import optic_flow
import numpy as np

speed = np.loadtxt("train.txt")
frame = 20400

input = np.zeros((100,))
output = np.zeros((100,))

n = 0
j = 0
while(n < 199):
    img1 = cv2.imread("stretched/%d.jpg" % n)
    img2 = cv2.imread("stretched/%d.jpg" % n+1)
    prev = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    diff = optic_flow(prev, next)
    input[j] = diff
    output[j] = np.mean(speed[n], speed[n+1])

model = make_model()

model.fit(x=input,
          y=output,
          batch_size=None,
          epochs=1,
          verbose=1)
