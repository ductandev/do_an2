import cv2
from matplotlib import pyplot as plt

img_BRG = cv2.imread("soccer.jpg",1)
img_GRAY = cv2.imread("soccer.jpg",0)

plt.imshow(img_BRG, cmap = 'gray', interpolation = 'bicubic')	# Ham doc anh, giong cv2.imread()
plt.xticks([]), plt.yticks([])					# danh dau toa do X, Y mac dinh
#plt.xticks([100,200]), plt.yticks([100,200])			# danh dau toa do X, Y
plt.title('ORIGINAL')						# Tieu de' hinh anh
plt.show()							# Hien thi 
#cv2.imwrite('a.png',img_BRG)

#cv2.waitKey(0)							# ham nay ko co tac dung voi matplotlip
#cv2.destroyAllWindows()					# ham nay ko co tac dung voi matplotlip
