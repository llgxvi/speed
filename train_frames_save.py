import cv2

cap = cv2.VideoCapture('train.mp4')

frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for i in range (frameC):
    ret, img = cap.read()

    if ret is False:
        print('Failure at frame', i)
        break

    cv2.imwrite('train_frames/%d.jpg' % i, img)

cap.release()
