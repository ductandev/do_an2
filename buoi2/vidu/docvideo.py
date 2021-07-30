import cv2
import numpy as np
cap = cv2.VideoCapture("/home/tan/Pictures/red_panda_snow.mp4")
while True:
    ret, frame = cap.read()
    resized = cv2.resize(frame, (720, 480))   ##########
    cv2.imshow("frame", resized)	      ##########
    key = cv2.waitKey(25)
    if key == 27:		#phim tat ESC = 27 
        break
#cv2.waitKey(0) 	 	#doi nhan 1 phim roi moi tat ha8n~
cap.release()
cv2.destroyAllWindows()

#############################################/*chinh sua kich thuoc hien thi imshow cua anh*/
#resized = cv2.resize(image, (200, 200))    #
#cv2.imshow("Fixed Resizing", resized)      #
#cv2.waitKey(0)                             #
#############################################
