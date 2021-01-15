import numpy as np
from cv2 import cvtColor
from cv2 import calcOpticalFlowFarneback as calc
from cv2 import cartToPolar
from cv2 import COLOR_RGB2GRAY, COLOR_RGB2HSV, COLOR_HSV2RGB
from cv2 import normalize, NORM_MINMAX

def optic_flow(img_rgb_curr, img_rgb_next):
    h, w, _ = img_rgb_curr.shape

    curr = cvtColor(img_rgb_curr, COLOR_RGB2GRAY)
    next = cvtColor(img_rgb_next, COLOR_RGB2GRAY)
    flow = calc(curr,
                next,
                None,
                pyr_scale=0.5,
                levels=1,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.3,
                flags=0)

    # magnitude, angle
    mag, ang = cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((h, w, 3))
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    hsv[:, :, 1] = cvtColor(img_rgb_next, COLOR_RGB2HSV)[:, :, 1]
    hsv[:, :, 2] = normalize(mag, None, 0, 255, NORM_MINMAX)

    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cvtColor(hsv, COLOR_HSV2RGB)

    return rgb_flow

if __name__ == '__main__':
    import cv2
    import sys

    i = int(sys.argv[1])

    curr = 'train_frames/%s.jpg' % i
    next = 'train_frames/%s.jpg' % (i + 1)

    curr = cv2.imread(curr)
    next = cv2.imread(next)

    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2RGB)

    cv2.imwrite('a.jpg', optic_flow(curr, next))
