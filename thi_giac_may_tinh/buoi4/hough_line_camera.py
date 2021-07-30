import numpy as np
import cv2 

cap = cv2.VideoCapture(0)		#('video.mp4')

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,450))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video mau xam

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength=50,maxLineGap=10)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('Video',frame)
    cv2.imshow('Canny',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
