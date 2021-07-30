import numpy as np
import cv2 

cap = cv2.VideoCapture(0)		#('video.mp4')

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video mau xam
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    cv2.imshow('Video_chia',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
