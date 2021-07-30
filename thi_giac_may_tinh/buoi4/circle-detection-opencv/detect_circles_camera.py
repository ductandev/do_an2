import numpy as np
import cv2 

cap = cv2.VideoCapture(0)		# Camera
#cap = cv2.VideoCapture('video.mp4')	# Video

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 75)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
             cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
             cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow('circles',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
