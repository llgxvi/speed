import numpy as np
import cv2

def change_brightness(img):
    x = 0.2 + np.random.uniform()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * x

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
