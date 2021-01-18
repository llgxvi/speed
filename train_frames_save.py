import cv2

from preprocess import preprocess
from progress import progress

cap = cv2.VideoCapture('train.mp4')

frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for i in range(frameC):
    ret, img = cap.read()

    if ret is False:
        print('Failure at frame', i)
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('train_frames/%d.jpg' % i, img)

    progress(i + 1, frameC)

cap.release()
