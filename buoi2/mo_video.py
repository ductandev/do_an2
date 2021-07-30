import numpy as np
import cv2
cap = cv2.VideoCapture('output.avi')
# cap = cv2.VideoCapture('output.avi')
# cap = cv2.VideoCapture('Minions_banana.mp4')
# Frame rate
fps = cap.get(cv2.CAP_PROP_FPS)  # 25.0
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# How many frames are there in total?
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('Total', num_frames, 'frame')
#
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print('high：', frame_height, 'width：', frame_width)
FRAME_NOW = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Frame 0
print('FRAME_NOW', FRAME_NOW)  # Current frame number 0.0
# Read the specified frame, valid for the video file, invalid for the camera?
frame_no = 121
cap.set(1, frame_no)  # Where frame_no is the frame you want
ret, frame = cap.read()  # Read the frame
cv2.imshow('frame_no'+str(frame_no), frame)
FRAME_NOW = cap.get(cv2.CAP_PROP_POS_FRAMES)
print('FRAME_NOW', FRAME_NOW)  # Current frame number 122.0
while cap.isOpened():
    ret, frame = cap.read()
    FRAME_NOW = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Current frame number
    print('FRAME_NOW', FRAME_NOW)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

