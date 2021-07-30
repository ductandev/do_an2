import cv2

img = cv2.imread("hinh.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
#thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

font = cv2.FONT_HERSHEY_COMPLEX

for cnt in contours:
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 8:
        cv2.putText(img, "8 canh", (x, y), font, 1, (0,255,0))
        img = cv2.drawContours(img, [approx], 0, (0,255,0), 2)
      
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
        cv2.putText(img, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 255), 2)
    

cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
