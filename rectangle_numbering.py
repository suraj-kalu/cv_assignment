import cv2

image = cv2.imread("image.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# cv2.imshow("Threshold Image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(contour) for contour in contours]
contour_areas = sorted(zip(areas, contours), reverse = True)

areas, contours = zip(*contour_areas)
line_contours = contours[-4:] #Select the last 4 smallest areas since they will be the areas if the 4 lines

for i, contour in enumerate(line_contours):
    m = cv2.moments(contour)
    
    #Compute the center
    x = int(m['m10']/m['m00'])
    y = int(m['m01']/m['m00'])
    
    #Assign the numbers
    cv2.putText(image, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

cv2.imshow("Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()