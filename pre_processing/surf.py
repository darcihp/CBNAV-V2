'''
import cv2
import numpy as np
img = cv2.imread("key.png", cv2.IMREAD_GRAYSCALE)

img = cv2.imread("4a4adae216e92063cabf04d9306a7694.png", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1500)

keypoints_orb, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import copy
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14.0, 7.0]    # Set figure_size size,Image display size (set the default image size)

varese_gray = cv2.imread('key.png')
varese_gray = cv2.cvtColor(varese_gray, cv2.COLOR_BGR2GRAY)

cv2.imshow('varese_gray',varese_gray)
cv2.waitKey(0)

varese = cv2.imread("4a4adae216e92063cabf04d9306a7694.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('varese',varese)

orb = cv2.ORB_create()

keypoints, descriptor = orb.detectAndCompute(varese_gray, None)

print(keypoints)

keyp_without_size = copy.copy(varese)
keyp_with_size = copy.copy(varese)

cv2.drawKeypoints(varese, keypoints, keyp_without_size, color = (0, 255,0 ))

cv2.drawKeypoints(varese, keypoints, keyp_with_size,
                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.subplot(121)
plt.title('2')
plt.imshow(keyp_without_size)


plt.subplot(122)
plt.title('1')
plt.imshow(keyp_with_size)
plt.show()


print("\nNumber of keypoints Detected: ", len(keypoints))
