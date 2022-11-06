import cv2
import numpy as np
from scipy import ndimage
image = cv2.imread("image.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
result = np.zeros(gray_image.shape)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

areas = [cv2.contourArea(contour) for contour in contours]
contour_areas = sorted(zip(areas, contours), reverse = True)
print("The sorted areas of the contours are {} ".format(sorted(areas, reverse = True)))

#Last 4 ares are the areas for the line while the last area is the area of the outermost countour. 
#For each rectangle the innermost and outermost contours are computed. So we will use only the outer contour.

rectangle_areas = contour_areas[1:9]
line_areas = contour_areas[-4:]

_, outer_rect_contours = zip(*rectangle_areas)
_, inner_line_contours = zip(*line_areas)

def copy_image(contours):
    angles = []
    contour_shape = np.zeros(gray_image.shape)
    for i, contour in enumerate(contours):
        min_rect = cv2.minAreaRect(contour)
        angle = int(min_rect[-1])
        epsilon = 1
        if angle not in angles and angle + epsilon not in angles and angle - epsilon not in angles:
            angles.append(angle)
            box = cv2.boxPoints(min_rect) #gives the 4 coordinates starting from the bottom left in a clockwise direction
            box = np.int0(box)
            temp = np.zeros(gray_image.shape)
            cv2.drawContours(temp, contour, -1, (255,255,255), 1)
            temp = ndimage.rotate(temp, angle)

            cv2.imwrite("temp.png", temp)
            temp1 = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)
            temp2 = cv2.resize(temp1, (gray_image.shape[1], gray_image.shape[0]))

            contour_shape = contour_shape + temp2
    return contour_shape

shape = copy_image(outer_rect_contours)
result = result + shape
shape = copy_image(inner_line_contours)
result = result + shape

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()