import cv2
import numpy as np
cap = cv2.VideoCapture("./videos/cat.mp4")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    averaging = cv2.blur(frame, (21, 21))
    cv2.imshow("frame", averaging)
    key = cv2.waitKey(25)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

