import cv2
import numpy as np
def nothing(x):
    pass
# Turns True when the mouse is pressed
drawing = False
# Draws a rectangle if mode is true. Press 'm' to change the curve. Mode=True
ix, iy = -1, -1
#Create a callback function
def draw_circle(event, x, y, flags, param):
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)
    global ix, iy, drawing, mode
    # When the left button is pressed, the coordinates of the starting position are returned.
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # When the left mouse button is pressed and moved, the drawing is drawn. Event can view the move, flag view is pressed
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                # Draw a circle, the dots are connected together to form a line, and 3 represents the thickness of the stroke.
                cv2.circle(img, (x, y), 3, color, -1)
                # The code commented out below is the starting point is the center of the circle, and the starting point to the end point is the radius.
                # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
                # cv2.circle(img,(x,y),r,(0,0,255),-1)
                # When the mouse is released, stop painting。
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # if mode==True:
        #     cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv2.circle(img,(x,y),5,(0,0,255),-1)
img = np.zeros((300, 512, 3), np.uint8)
mode = False
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)  # & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == ord("q"):
        break

