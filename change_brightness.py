import cv2

def change_brightness(img, bright_factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * bright_factor

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
