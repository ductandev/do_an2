import numpy as np
import matplotlib.pyplot as plt

w=10
h=10
fig=plt.figure(figsize=(5, 5))		# plt.figure:Tao cua so hien thi anh, figsize :(chiều rộng, chiều cao) tính bằng inch
columns = 4
rows = 5
for i in range(1, columns*rows +1):		# i chay tu 1 den 21
    img = np.random.randint(10, size=(h,w))	# random. so nguyen (tu 0-10, co kichh thuoc dai,rong =10)
    fig.add_subplot(rows, columns, i)		# Them hinh anh (tong hang,tong cot, nam o vi tri')
    plt.imshow(img)				# Ham doc anh, giong cv2.imread()
plt.show()					# hiển thị tất cả các Du liệu
