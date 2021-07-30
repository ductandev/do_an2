import cv2  # Not actually necessary if you just want to create an image.
import numpy as np

blank_image = np.zeros((400,400,3), np.uint8)
cv2.line(blank_image, (50,50), (50,50), (255,255,255), 10)
cv2.imshow('image1',blank_image)

blank_image[:,:,:] = (255,0,0)      # (B, G, R)
cv2.line(blank_image, (0,0), (200,300), (0,0,255), 5)
cv2.imshow('image2',blank_image)

blank_image[:,400//2:400] = (0,255,0)
cv2.line(blank_image, (0,0), (200,300), (0,0,255), 5)
cv2.imshow('image3',blank_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

