import cv2
import numpy as np

def nothing(x):
	pass
cap=cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
	_,frame=cap.read()
	#frame=cv2.flip(frame,1)
	hsv_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#print(hsv_frame)
	l_h=cv2.getTrackbarPos("L-H", "Trackbars")
	l_s=cv2.getTrackbarPos("L-S", "Trackbars")
	l_v=cv2.getTrackbarPos("L-V", "Trackbars")
	u_h=cv2.getTrackbarPos("U-H", "Trackbars")
	u_s=cv2.getTrackbarPos("U-S", "Trackbars")
	u_v=cv2.getTrackbarPos("U-V", "Trackbars")

	#min_mau=np.array([l_h,l_s,l_v])
	#max_mau=np.array([u_h,u_s,u_v])
	min_mau=np.array([78,60,78])
	max_mau=np.array([255,255,255])
	mask=cv2.inRange(hsv_frame,min_mau,max_mau)
	
	result=cv2.bitwise_and(frame,frame,mask=mask)

	#print(result)
	#i=206
	#j=276
	#count=0
	#for i in range(469):
		#for j in range(425):
			#if(0<=result[i,j,0]<=255):
				#if(170<=result[i,j,1]<=255 & 150<=result[i,j,2]<=255):
					#count+=1
					#if(i==469 & j==425):
						#i=206
						#j=276
			#print(count)				

	cv2.imshow("frame",hsv_frame)
	cv2.imshow("mask",mask)
	cv2.imshow("result",result)
	key = cv2.waitKey(1)
	if key==27:
		break
cap.release()
cv2.destroyAllWindows()
	

