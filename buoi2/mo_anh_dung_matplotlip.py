import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('soccer.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('a.png',img)
cv2.destroyAllWindows()
