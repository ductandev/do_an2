import cv2
import numpy as np

img = cv2.imread('road.jpg',0)		#chọn đường dẫn ảnh để đọc
img = cv2.resize(img,(250,200))
#------------------------------------------------------
averaging = cv2.blur(img, (21, 21))
gaussian = cv2.GaussianBlur(img, (21, 21), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 350, 350)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)
erosion = cv2.erode(img, kernel, iterations=6)
img1 = cv2.GaussianBlur(img, (11, 11), 0)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
#-----------------------------------------------------------------------------------------------------------------------------------------

cv2.imshow('averaging',averaging)
cv2.imshow('gaussian',gaussian)			
cv2.imshow('median',median)
cv2.imshow('bilateral',bilateral)
cv2.imshow('GaussianBlur',img1)		
cv2.imshow('laplacian',laplacian)	
cv2.imshow('dilation',dilation)	
cv2.imshow('erosion',erosion)		
cv2.waitKey(0)				# tạm dừng cho đến khi tắt 
cv2.destroyAllWindows()			# giải phóng bộ nhớ

