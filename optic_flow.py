import numpy as np
from cv2 import cvtColor
from cv2 import calcOpticalFlowFarneback as calc
from cv2 import cartToPolar
from cv2 import COLOR_RGB2GRAY, COLOR_RGB2HSV, COLOR_HSV2RGB
from cv2 import normalize, NORM_MINMAX

def optic_flow(img_curr, img_next):
    h, w, _ = img_curr.shape

    curr = cvtColor(img_curr, COLOR_RGB2GRAY)
    next = cvtColor(img_next, COLOR_RGB2GRAY)
    flow = calc(curr,
                next,
                None,
                0.5,
                1,
                15,
                2,
                5,
                1.3,
                0)

    # magnitude, angle
    mag, ang = cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((h, w, 3))
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    hsv[:, :, 1] = cvtColor(img_next, COLOR_RGB2HSV)[:, :, 1]
    hsv[:, :, 2] = normalize(mag, None, 0, 255, NORM_MINMAX)
    
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cvtColor(hsv, COLOR_HSV2RGB)

    return rgb_flow

if __file__ == '__main__':
    pass
    
