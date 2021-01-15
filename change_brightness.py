import cv2

def change_brightness(img_rgb, bright_factor):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * bright_factor

    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_rgb
