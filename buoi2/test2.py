import numpy as np
import cv2
from matplotlib import pyplot as plt
cap = cv2.VideoCapture('output.avi')
while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	plt.imshow(gray, cmap = 'gray', interpolation = 'bicubic')
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
cap.release()
cv2.destroyAllWindows()
