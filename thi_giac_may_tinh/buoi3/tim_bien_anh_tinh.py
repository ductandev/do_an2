import cv2

img = cv2.imread('road.jpg',0)		#chọn đường dẫn ảnh để đọc
img = cv2.resize(img,(300,250))
#-----------------------------------------------------
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
canny = cv2.Canny(img, 100, 150)
canny1 = cv2.Canny(img,50,200)		#hàm đọc biên ảnh (đường dẫn ảnh, độ sáng ngưỡng ảnh muốn đọc trong khoản từ nhỏ nhất là  50 đến lớn nhất là 200)
#-----------------------------------------------------------------------------------------------------------------------------------------

cv2.imshow('sobelx',sobelx)		# hiển thị ảnh
cv2.imshow('sobely',sobely)		# hiển thị ảnh
cv2.imshow('laplacian',laplacian)		# hiển thị ảnh
cv2.imshow('canny',canny)		# hiển thị ảnh
cv2.imshow('canny1',canny1)		# hiển thị ảnh

cv2.waitKey(0)				# tạm dừng cho đến khi tắt 
cv2.destroyAllWindows()			# giải phóng bộ nhớ

