import numpy as np
import cv2
import random as rng

class canny_image:
    def __init__(self,_image,):
        self.image = _image

    def apply_canny(self, _sigma=0.33):
        image = cv2.imread("./"+self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #v = np.median(blurred)
        #lower = int(max(0, (1.0 - _sigma) * v))
        #upper = int(min(255, (1.0 + _sigma) * v))
        #edged = cv2.Canny(image, lower, upper)
        #contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        cX = [None]*len(contours)
        cY = [None]*len(contours)

        for i, c in enumerate(contours):
        	contours_poly[i] = cv2.approxPolyDP(c, 1, True)
        	#contours_poly[i] = cv2.convexHull(c)

        	boundRect[i] = cv2.boundingRect(contours_poly[i])
        	M = cv2.moments(c)
        	if M["m00"] != 0:
        		cX[i] = int(M["m10"] / M["m00"])
        		cY[i] = int(M["m01"] / M["m00"])
        	else:
        		cX[i] = 0
        		cY[i] = 0
        square_id = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','Z']

        for i in range(len(contours)):
        	color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        	cv2.drawContours(image, contours_poly, i, (0,0,255), -1)
        	cv2.putText(image, square_id[i], (cX[i], cY[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image

ci = canny_image("4a4adae216e92063cabf04d9306a7694.png")

def on_change_erode(value):
    _pre_image.set_erode(value)
    cv2.imshow(windowName, _pre_image.find_contours(False))

def on_change_dilate(value):
    _pre_image.set_dilate(value)
    cv2.imshow(windowName, _pre_image.find_contours(False))

def on_change_blur(value):
    _pre_image.set_blur(value)
    cv2.imshow(windowName, _pre_image.find_contours(False))

windowName = 'image'
trackErode = 'trackErode'
trackDilate = 'trackDilate'
trackBlur = 'trackBlur'

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.createTrackbar(trackErode, windowName, 0, 255, on_change_erode)
cv2.createTrackbar(trackDilate, windowName, 0, 255, on_change_dilate)
cv2.createTrackbar(trackBlur, windowName, 0, 5000, on_change_blur)

cv2.imshow(windowName, ci.apply_canny()) 

while True:
    # Press Esc to exit
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()