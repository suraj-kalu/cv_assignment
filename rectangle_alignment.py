import cv2
import numpy as np
from scipy import ndimage
image = cv2.imread("image.png")

cX = image.shape[0] // 2
cY = image.shape[1] // 2

img1 = image[0:cX, 0 :cY]
img2 = image[cX: image.shape[0], 0:cY]
img3 = image[0:cX,cY:image.shape[1]]
img4 = image[cX: image.shape[0], cY:image.shape[1]]

for image in [img1, img2, img3, img4]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours[1], -1, (0,255,0), 2)
    rect = cv2.minAreaRect(contours[1])
    image = ndimage.rotate(image, rect[-1])
    cv2.imshow("image" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()