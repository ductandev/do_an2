import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret = cap.set(3, 640)
ret = cap.set(4, 480)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_blue = np.array([38, 86, 0])
upper_blue = np.array([121, 255, 255])
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask1 = cv2.inRange(hsv, lower_yellow, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.bitwise_and(frame, frame, mask=mask1)

    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', x=0, y=0)
    cv2.imshow('mask', mask)
    cv2.moveWindow('mask', x=frame.shape[1], y=0)
    cv2.imshow('res', res)
    cv2.moveWindow('res', y=frame.shape[0], x=0)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

