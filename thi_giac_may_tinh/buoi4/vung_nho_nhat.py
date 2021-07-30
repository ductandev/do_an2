import numpy as np
import cv2

img = cv2.imread('2.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
