import cv2 as cv

def change_brightness(img, bf):
    '''
    img:    bgr
    bf:     0~1
    return: bgr
    '''

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * bf

    bgr  = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr
