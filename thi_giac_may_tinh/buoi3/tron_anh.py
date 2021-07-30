import cv2
import numpy as np
from matplotlib import pyplot as plt
img1 = cv2.imread('road.jpg')
img1 = cv2.resize(img1,(1280,720))
img2 = cv2.imread('xe.jpg')
dst = cv2.addWeighted(img1,0.7,img2,1,0)
dst = cv2.resize(dst,(720,480))
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
