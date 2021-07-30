import cv2
import numpy as np

img=np.zeros((500,500,3), np.uint8)

cv2.ellipse(img, (250,100), (70,70), 135, 0, 270, (128,0,255),50)
cv2.ellipse(img, (150,280), (70,70), 10, 0, 270, (0,255,0),50)
cv2.ellipse(img, (350,280), (70,70), 315, 0, 270, (255,0,0),50)
cv2.ellipse(img,(200,160),(30,30),315,0,180,(0,0,0),-1)
cv2.ellipse(img,(290,160),(30,30),45,0,180,(0,0,255),-1)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(75,450), font, 3,(255,255,255),2,cv2.LINE_AA)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
