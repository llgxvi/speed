import cv2 as cv

def preprocess(img):
    '''
    img:    bgr
    return: bgr
    '''

    img = img[100:-100, 100:-100]

    img = cv.resize(img, (200, 100), interpolation=cv.INTER_AREA)

    return img
