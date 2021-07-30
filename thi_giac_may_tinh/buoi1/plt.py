import cv2
import matplotlib.pyplot as plt
import numpy as np

imgLr = cv2.imread("soccer.jpg",1)
imgRr = cv2.imread("logo.png",1)

f = plt.figure()			# Tao mot hinh anh, hinh ve minh hoa, sơ đồ mới
f.add_subplot(1,2, 1)			# Them hinh anh (tong hang,tong cot, nam o vi tri')
#plt.imshow(np.rot90(imgLr,2))		# xoay 90'
plt.imshow(imgLr)			# Ham doc anh, giong cv2.imread()
f.add_subplot(1,2, 2)			# Them hinh anh (tong hang,tong cot, nam o vi tri')
#plt.imshow(np.rot90(imgRr,2))		# xoay 90'
plt.imshow(imgRr)			# Ham doc anh, giong cv2.imread()
plt.show(block=True)			# hiển thị tất cả các Du liệu
