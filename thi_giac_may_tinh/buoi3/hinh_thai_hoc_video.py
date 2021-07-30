import numpy as np
import cv2

#cap = cv2.VideoCapture(0)		#('video.mp4')
cap = cv2.VideoCapture("los_angeles.mp4")		#('video.mp4')

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(400,250))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video mau xam

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(frame, kernel)
    dilation = cv2.dilate(frame, kernel, iterations=1)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('dilation', dilation)
    cv2.imshow('erosion', erosion)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    #cv2.moveWindow('dilation', x=frame.shape[1], y=0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()
