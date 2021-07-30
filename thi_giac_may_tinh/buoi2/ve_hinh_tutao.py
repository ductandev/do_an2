import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
#blank_image = np.zeros((450,450,3), np.uint8)
blank_image = cv2.imread('thuysinh.jpg') 
cv2.line(blank_image, (0,0), (400,300), (0,0,255), 5)
cv2.line(blank_image, (200,40), (200,40), (255,255,255), 20)
cv2.rectangle(blank_image,(20,20),(400,300),(0,255,0),3)
cv2.circle(blank_image,(90,100), 60, (0,0,255), -1)
cv2.line(blank_image, (0,0), (400,300), (0,0,255), 5)
cv2.line(blank_image, (500,500), (500,500), (0,0,255), 20)
cv2.ellipse(blank_image,(256,256),(50,25),0,0,360,255,-1)
#cv2.polylines(blank_image,[pts],True,(0,255,255))
pts = np.array([[25, 200], [400, 200], 
		[360, 250], [300, 10], 
		[200, 250], [110, 100]], np.int32) 

pts = pts.reshape((-1, 1, 2)) 
cv2.polylines(blank_image,[pts],True,(0,255,255),5)
cv2.imshow('image3',blank_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

