import numpy as np
import cv2 as cv

def optic_flow(curr, next):
    '''
    curr:   bgr
    next:   bgr
    return: bgr
    '''

    gray1 = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(
                gray1,
                gray2,
                None,
                pyr_scale=0.5,
                levels=1,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.1,
                flags=0)

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros(curr.shape)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    hsv = hsv.astype(np.float32)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr
