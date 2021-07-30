import cv2
import numpy as np

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # lower_red = np.array([161, 155, 84]) #mau hong nua
    # upper_red = np.array([179, 255, 255]) #mau hong nua

    # lower_blue = np.array([101, 50, 38])
    # upper_blue = np.array([110, 255, 255])

    # mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
