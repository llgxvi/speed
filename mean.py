import numpy as np

speed = np.loadtxt('train.txt')

a = np.zeros((20399))

for i in range(20400-1):
    a[i] = np.absolute(speed[i+1] - speed[i])

print(a.shape)
print(a[:10])
print(np.mean(a))
