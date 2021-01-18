import cv2

# rgb: rgb img
# bf:  bright factor
def change_brightness(rgb, bf):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * bf

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
