import cv2 
import numpy as np 

image = cv2.imread('thuysinh.jpg') 
window_name = 'Image'

#pts = np.array([[25, 200], [400, 200], 
#		[360, 250], [300, 10], 
#		[200, 250], [110, 100]], np.int32)
pts = np.array([[25, 200], [150, 200], 
		[360, 250], [300, 10], 
		[200, 250], [110, 100]], np.int32) 

pts = pts.reshape((-1, 1, 2)) 
isClosed = True
color = (255, 0, 0) 
thickness = 2

image = cv2.polylines(image, [pts], isClosed, color, thickness) 

while(1): 
	
	cv2.imshow('image', image) 
	if cv2.waitKey(20) & 0xFF == 27: 
		break
		
cv2.destroyAllWindows() 

