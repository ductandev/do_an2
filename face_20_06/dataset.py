# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os


#path = "/home/tan/AI/giao dien Tkinter/test/tan"

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(src='rtsp://192.168.1.100:554/h264&basic_auth=YWRtaW46MTIzNA==').start()
time.sleep(2.0)
total = 0
detector = cv2.CascadeClassifier("/home/tan/AI/giao dien Tkinter/test/haarcascade_frontalface_default.xml") #(lỗi -215: là lỗi ko tìm thấy file haar đường dẫn này )
while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		p = os.path.sep.join([path, "anh_{}.png".format(str(total).zfill(1))])
		cv2.imwrite(p, orig)
		total += 1
		time.sleep(1)
		#cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", image_frame[y:y+h,x:x+w])
	# show the output frame
	cv2.imshow("Frame", frame)
	if cv2.waitKey(50) & 0xFF == ord('q'):
		break
	# If image taken reach 100, stop taking video
	elif (total >= 7):
		break
	# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
vs.stop()
cv2.destroyAllWindows()
