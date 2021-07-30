import cv2
import numpy as np
img = cv2.imread('soccer.jpg', 0)
cv2.imshow('j.png', img)
print(img.shape)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('dilation', dilation)
cv2.moveWindow('dilation', x=img.shape[1], y=0)
cv2.waitKey(0)
cv2.destroyAllWindows()
