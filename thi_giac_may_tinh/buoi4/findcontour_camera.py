import numpy as np
import cv2 

cap = cv2.VideoCapture(0)		#('video.mp4')

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video mau xam
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of contours = " + str(len(contours)))
    print(contours[0])

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Image', frame)
    cv2.imshow('Image GRAY', imgray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
