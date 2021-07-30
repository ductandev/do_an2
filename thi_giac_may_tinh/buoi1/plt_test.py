import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("cat.mp4",1)
#img1 = cv2.imread("soccer.jpg",1)
img2 = cv2.imread("logo.png",1)

f = plt.figure()						# Tao mot hinh anh, hinh ve minh hoa, sơ đồ mới

f.add_subplot(1,2, 1)						# Them hinh anh (tong hang,tong cot, nam o vi tri')
a = plt.imshow(img1, cmap = 'gray', interpolation = 'bicubic')	# Ham doc anh, giong cv2.imread()
#plt.xticks([]), plt.yticks([])					# danh dau toa do X, Y mac dinh
plt.title('soccer')						# Tieu de' hinh anh

f.add_subplot(1,2, 2)						# Them hinh anh (tong hang,tong cot, nam o vi tri')
b = plt.imshow(img2)						# Ham doc anh, giong cv2.imread()
#plt.xticks([]), plt.yticks([])					# danh dau toa do X, Y mac dinh
plt.title('logo opencv')					# Tieu de' hinh anh

plt.savefig('save_anh.png')

plt.show()							# Hien thi tat ca du lieu
plt.close(f)




