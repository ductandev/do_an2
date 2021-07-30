import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('digits.png', 0)

cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]
print (cells[0][0])
cv2.imwrite('anhketqua.jpg',cells[0][0])
cv2.waitKey(0)
cv2.destroyAllWindows()
