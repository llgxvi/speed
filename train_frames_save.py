import cv2 as cv
from progress import progress

cap = cv.VideoCapture('train.mp4')

count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

for i in range(count):
    ret, img = cap.read()

    if ret is False:
        print('Failure at frame', i)
        break

    cv.imwrite('train_frames/%d.jpg' % i, img[190:-190, 220:-220])

    progress(i + 1, count)

cap.release()
