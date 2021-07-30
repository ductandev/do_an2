import cv2
import numpy as np

grayscaled = cv2.imread('logo.png',0)
grayscaled = cv2.resize(grayscaled,(400,400))

retval2,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th=cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 115, 1)

cv2.imshow('original', grayscaled)
cv2.imshow('Otsu threshold',threshold2)
cv2.imshow('test',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
