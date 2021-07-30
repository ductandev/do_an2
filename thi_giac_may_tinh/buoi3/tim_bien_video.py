import cv2
import numpy as np
#cap = cv2.VideoCapture("los_angeles.mp4")
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame,(300,200))
#--------------------------------------------------------
	sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
	laplacian = cv2.Laplacian(frame, cv2.CV_64F, ksize=5)
	canny = cv2.Canny(frame, 100, 150)
	canny1 = cv2.Canny(frame,50,200)		#hàm đọc biên ảnh (đường dẫn ảnh, độ sáng ngưỡng ảnh muốn đọc trong khoản từ nhỏ nhất là  50 đến lớn nhất là 200)
#--------------------------------------------------------
	cv2.imshow("sobelx", sobelx)
	cv2.imshow("sobely", sobely)
	cv2.imshow("laplacian", laplacian)
	cv2.imshow("canny", canny)
	cv2.imshow("canny1", canny1)
	key = cv2.waitKey(25)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()

