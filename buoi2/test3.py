import cv2
from skimage.io import imread
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt

cap = cv2.VideoCapture('output.avi') 
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    count +=1  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray =  scipy.misc.imresize(gray, 0.45) 

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])

    cv2.imshow('Gray', gray)
    cv2.waitKey(5)
    plt.title("Histogram")

    plt.plot(hist)
    plt.show()

cap.release()
cv2.destroyAllWindows()
