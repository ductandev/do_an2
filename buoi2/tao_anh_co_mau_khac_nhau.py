import cv2
import numpy as np

blank_image = np.zeros((400,400,3), np.uint8)
cv2.imshow('image1',blank_image)

blank_image[:,:,:] = (255,0,0)      # (B, G, R)
cv2.imshow('image2',blank_image)

blank_image[:,400//2:400] = (0,255,0)
cv2.imshow('image3',blank_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

