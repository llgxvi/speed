import cv2

def imread(index):
    img = cv2.imread('train_frames/' + str(index) + '.jpg')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb
