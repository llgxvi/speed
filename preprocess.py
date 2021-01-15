import cv2

def preprocess(img_rgb, bright_factor=0):
    # crop
    img = img_rgb[100:300, 100:440]

    # resize
    img = cv2.resize(img, (200, 60), interpolation=cv2.INTER_AREA)

    # augment saturation
    if bright_factor > 0:
        img = change_brightness(img, bright_factor)

    return img # rgb
