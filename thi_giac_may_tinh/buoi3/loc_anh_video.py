import cv2
import numpy as np
cap = cv2.VideoCapture("los_angeles.mp4")
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame,(300,200))
#--------------------------------------------------------
	gaussian = cv2.GaussianBlur(frame, (21, 21), 0)
	averaging = cv2.blur(frame, (21, 21))
	median = cv2.medianBlur(frame, 5)
	bilateral = cv2.bilateralFilter(frame, 9, 350, 350)

	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(frame,kernel,iterations = 1)
	erosion = cv2.erode(frame, kernel, iterations=6)
#--------------------------------------------------------
	cv2.imshow("averaging", averaging)
	cv2.imshow("gaussian", gaussian)
	cv2.imshow("median", median)
	cv2.imshow("bilateral", bilateral)
	cv2.imshow("dilation", dilation)
	cv2.imshow("erosion", erosion)
	key = cv2.waitKey(25)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()

