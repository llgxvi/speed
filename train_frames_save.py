import cv2 as cv
from progress import progress
from preprocess import preprocess

cap = cv.VideoCapture('train.mp4')

count = cap.get(cv.CAP_PROP_FRAME_COUNT)
count = int(count)

for i in range(count):
    ret, img = cap.read()

    img = preprocess(img)

    cv.imwrite('train_frames/%d.jpg' % i, img)

    progress(i + 1, count)

cap.release()
