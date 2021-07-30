import cv2
#import numpy as np

#img = np.uint8([[[232,162,0]]])	# tao bo loc de loc anh (B, R G)
#hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#print (hsv_img)

img2=cv2.imread('anh1.jpg',1)
resized_goc = cv2.resize(img2, (400, 350))
cv2.imshow("anh goc",resized_goc)

img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.resize(img_gray, (400, 350))
cv2.imshow('anh xam',resized_gray)

cv2.waitKey(0)
