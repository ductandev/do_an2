import numpy as np
import cv2

size = (2560, 1600)
# All black. Can be used in screensavers
black = np.zeros(size)
print(black[34][56])
cv2.imwrite('black.jpg',black)
#White all white
black[:]=255
print(black[34][56])
cv2.imwrite('white.jpg',black)

