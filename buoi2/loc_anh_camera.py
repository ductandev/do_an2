import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    gaussian = cv2.GaussianBlur(frame, (21, 21), 0)
    cv2.imshow("frame", gaussian)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

