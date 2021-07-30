import cv2

img = cv2.imread('soccer.jpg',0)		#chọn đường dẫn ảnh để đọc
#------------------------------------------------------
averaging = cv2.blur(img, (21, 21))
gaussian = cv2.GaussianBlur(img, (21, 21), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 350, 350)
#dilation = cv2.dilate(mask, kernel)
#erosion = cv2.erode(mask, kernel, iterations=6)
img1 = cv2.GaussianBlur(img, (11, 11), 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
canny = cv2.Canny(img, 100, 150)
#-----------------------------------------------------
img2 = cv2.Canny(img,50,200)		#hàm đọc biên ảnh (đường dẫn ảnh, độ sáng ngưỡng ảnh muốn đọc trong khoản từ nhỏ nhất là  50 đến lớn nhất là 200)
cv2.imshow('bienanh',sobely)		# hiển thị ảnh
cv2.waitKey(0)				# tạm dừng cho đến khi tắt 
cv2.destroyAllWindows()			# giải phóng bộ nhớ

