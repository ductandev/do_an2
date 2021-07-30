import cv2
import numpy as np
cap = cv2.VideoCapture("/home/tan/Pictures/red_panda_snow.mp4")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("flipped_red_panda.avi", fourcc, 25, (640, 360))
while True:
    ret, frame = cap.read()
    frame2 = cv2.flip(frame, 1)		#ham nguoc video(x,y)
    cv2.imshow("frame2", frame2)
    cv2.imshow("frame", frame)
    out.write(frame2)			#luu video frame2
    key = cv2.waitKey(25)
    if key == 27:			#ESC thoat chuong trinh
        break
out.release()
cap.release()
cv2.destroyAllWindows()
