import cv2

img_BRG = cv2.imread("soccer.jpg",1)
img_GRAY = cv2.imread("soccer.jpg",0)


cv2.imshow("anh goc",img_BRG)
cv2.imshow("anh xam",img_GRAY)

cv2.imwrite("anh_da_luu.jpg",img_GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()
