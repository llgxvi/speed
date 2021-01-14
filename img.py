import cv2
import sys
from change_brightness import change_brightness

cap = cv2.VideoCapture(sys.argv[1] + '.mp4')
frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for c in range (frameC):
    ret, img = cap.read()

    if ret is False:
        print('Fail at frame', c)
        break

    img = img[150:-150, 200:-200, :]
    img = change_brightness(img)
    cv2.imwrite('frame_%s/%d.jpg' % (sys.argv[1], c), img)

cap.release()

if __name__ == '__main__':
    im = cv2.imread('frame_train/0.jpg')
    print(im.shape, im.dtype, type(im))
