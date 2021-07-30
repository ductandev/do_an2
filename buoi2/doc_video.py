import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('test.mp4', fourcc, 20.0, (640,480))
#print(cap.isOpened()) #in ra do phan giai (dai x rong)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
	plt.xticks([]), plt.yticks([])
	cv2.imshow('gray', gray)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640,480))

	if cv2.waitKey(1) & 0xFF == ord('q'):
	   break
cap.release()
out.release()
cv2.destroyAllWindows()
