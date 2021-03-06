import cv2
import numpy as np 

img = cv2.imread("2.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)
w, h,_ = template.shape[::-1]

res = cv2.matchTemplate(img1, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.35
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,0,255), 2)
cv2.imshow("detected", img)

k= cv2.waitKey(5) & 0xFF
if k==27:
    cv2.destroyAllWindows()
