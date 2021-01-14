import cv2

cap = cv2.VideoCapture('train.mp4')
frameC = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for c in range (frameC):
    ret, img = cap.read()

    if ret is False:
        print('Fail at frame', c)
        break

    img = img[150:-150, 200:-200, :]
    cv2.imwrite('frame/%d.jpg' % c, img)

cap.release()

if __name__ == '__main__':
    im = cv2.imread('frame/0.jpg')
    print(im.shape, im.dtype, type(im))
