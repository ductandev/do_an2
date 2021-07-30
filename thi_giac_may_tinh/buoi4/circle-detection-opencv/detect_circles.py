import numpy as np
import argparse
import cv2

image = cv2.imread('images/8circles.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 75)
print(circles)
if circles is not None:
	circles = np.round(circles[0, :]).astype("int")	 # ham round: lam tron so, int: ra kieu so nguyen
	print(circles)	#in ra (X, Y, ban kinh)
	for (x, y, r) in circles:
		cv2.circle(image, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	cv2.imshow("output", image)
	cv2.waitKey(0)
