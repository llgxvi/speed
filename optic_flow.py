import numpy as np
from cv2 import cvtColor
from cv2 import calcOpticalFlowFarneback as calc
from cv2 import cartToPolar
from cv2 import COLOR_RGB2GRAY, COLOR_RGB2HSV, COLOR_HSV2RGB
from cv2 import normalize, NORM_MINMAX

def optic_flow(img_curr, img_next):
    gray_curr = cvtColor(img_curr, COLOR_RGB2GRAY)
    gray_next = cvtColor(img_next, COLOR_RGB2GRAY)

    hsv = np.zeros((100, 200, 3))
    hsv[:,:,1] = cvtColor(img_next, COLOR_RGB2HSV)[:,:,1]

    flow = calc(gray_curr,
                gray_next,
                None,
                0.5,
                1,
                15,
                2,
                5,
                1.3,
                0)

    mag, ang = cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:,:,0] = ang * (180 / np.pi / 2)
    hsv[:,:,2] = normalize(mag,None,0,255,NORM_MINMAX)
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cvtColor(hsv,COLOR_HSV2RGB)

    return rgb_flow
