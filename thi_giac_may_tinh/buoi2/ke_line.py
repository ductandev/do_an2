import numpy as np
import cv2

img = cv2.imread("thuysinh.jpg", 1)
cv2.line(img, (0,0), (400,300), (0,0,255), 5)
cv2.line(img, (500,500), (500,500), (0,0,255), 20)
#cv2.imwrite('dave.jpg', img);

cv2.imshow('img' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()
