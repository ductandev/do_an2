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

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        (x, y, w, h) = cv2.boundingRect(cnt)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        # x = approx.ravel()[0]
        # y = approx.ravel()[1]

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)
            cv2.line(frame, (x_medium, y_medium), (x_medium, y_medium), (255, 255, 255), 4)

            if len(approx) == 3:
                cv2.putText(frame, "tam giac", (x, y), font, 1, (0, 0, 255))
            elif len(approx) == 4:
                cv2.putText(frame, "hinh vuong", (x, y), font, 1, (0, 0, 255))
            elif 10 <= len(approx) :
                cv2.putText(frame, "hinh tron", (x, y), font, 1, (0, 0, 255))


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
