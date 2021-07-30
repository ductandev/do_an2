import numpy as np
import cv2

img=cv2.imread('dining_table.jpg',1)
cap = cv2.VideoCapture(0)		#('video.mp4')

while(True):
    ret, frame = cap.read()
    frame1 = cv2.resize(frame,(500,500))
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video mau xam

    cong = cv2.add(img,frame1)
    tru = cv2.subtract(img, frame1)
    nhan = cv2.multiply(img, frame1)
    chia = cv2.divide(img, frame1)

    cv2.imshow('Video_cong',cong)
    cv2.imshow('Video_tru',tru)
    cv2.imshow('Video_nhan',nhan)
    cv2.imshow('Video_chia',chia)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()
