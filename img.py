import cv2

# extract video frame, save as jpg
# h,w
# 480,640
# 100,150 cropped
# 100,200 stretched

cap = cv2.VideoCapture('train.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

c = 0
ret = True
while (c < frameCount and ret):
    ret, img = cap.read()

    # frame
    cv2.imwrite("train/%d.jpg" % c, img)

    # cropped
    img = img[189:189+100, 244:244+150, :]
    cv2.imwrite("cropped/%d.jpg" % c, img)

    # stretched
    img = cv2.resize(img, dsize=(200, 100), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("stretched/%d.jpg" % c, img)

    c += 1

cap.release()
