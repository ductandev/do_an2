from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter import *							#####
from PIL import Image, ImageTk					#####
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import abl
import dfa
import os
import time
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import imutils

def exit_click():									# Cú pháp Button phải đặt dưới vòng def exit_btn() thì button mới nhảy vào trong được	
	print("Clode Window slave success !")
	fps.stop()                                                                                                                              
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	video_capture.release()
	root.destroy()									# chỉ tắt cửa sổ con


def show_frame():
	global c
	global img_counter
	global sss
	global xxx
	global zzz
	global vvv

	ret, frame = video_capture.read()

	frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #resize frame (optional)

	curTime = time.time()+1    # calc fps
	timeF = frame_interval

	if (c % timeF == 0): 
		c = 0
		find_results = []

		if frame.ndim == 2:
			frame = abl.to_rgb(frame)
		frame = frame[:, :, 0:3]
		bounding_boxes, _ = dfa.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
		nrof_faces = bounding_boxes.shape[0]
		#print('Detected_FaceNum: %d' % nrof_faces)

		if nrof_faces > 0:
			det = bounding_boxes[:, 0:4]
			img_size = np.asarray(frame.shape)[0:2]

			cropped = []
			scaled = []
			scaled_reshape = []
			bb = np.zeros((nrof_faces,4), dtype=np.int32)
			k = 0
			for i in range(nrof_faces):
				emb_array = np.zeros((1, embedding_size))

				bb[k][0] = det[i][0]
				bb[k][1] = det[i][1]
				bb[k][2] = det[i][2]
				bb[k][3] = det[i][3]

				# inner exception
				if bb[k][0] <= 0 or bb[k][1] <= 0 or bb[k][2] >= len(frame[0]) or bb[k][3] >= len(frame):
					print('Face is very close!')
					continue

				cropped.append(frame[bb[k][1]:bb[k][3], bb[k][0]:bb[k][2], :])
				cropped[k] = abl.flip(cropped[k], False)
				scaled.append(misc.imresize(cropped[k], (image_size, image_size), interp='bilinear'))
				scaled[k] = cv2.resize(scaled[k], (input_image_size,input_image_size),
									interpolation=cv2.INTER_CUBIC)
				scaled[k] = abl.prewhiten(scaled[k])
				scaled_reshape.append(scaled[k].reshape(-1,input_image_size,input_image_size,3))
				feed_dict = {images_placeholder: scaled_reshape[k], phase_train_placeholder: False}
				emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
				predictions = model.predict_proba(emb_array)
	#			print(predictions)
				best_class_indices = np.argmax(predictions, axis=1)
				best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
				# print("predictions")
				#print(best_class_indices,' with accuracy ',best_class_probabilities)

				# print(best_class_probabilities)
				result_names = names[best_class_indices[0]]
				if (best_class_probabilities >= 0.16) and (result_names != 'unknown'):
					cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
					time_open_door = 0
					percent = best_class_probabilities[0] * 100 + 40
					if(percent > 100):
						percent = 100
					if 1:
						max = predictions[0][0]
						for a in predictions[0]:
							if(a > max):
								max = a
						max2 = predictions[0][0]
						cnt_2 = 0
						flag = 0
						for a in predictions[0]:
							if(a > max2 and a < max):
								flag = cnt_2
								max2 = a
							cnt_2 = (cnt_2 + 1)
					#------------------------------------------
					#cv2.imwrite("hinh/anh_%d.png"%img_counter,frame)
					#img = Image.open("hinh/anh_%d.png"%img_counter)
					#img = img.resize((250,250), Image.ANTIALIAS)
					#photoImg =  ImageTk.PhotoImage(img)			# dùng biến này tiếp tục in cho biến 2
					#show_photo_label.image = photoImg
					#show_photo_label.configure(image=photoImg)
					
					#if img_counter-5 >= 0 :
						#img1 = Image.open("hinh/anh_%d.png"%(img_counter-5))
						#img1 = img1.resize((250,250), Image.ANTIALIAS)
						#photoImg1 =  ImageTk.PhotoImage(img1)
						#show_photo_label_1.image = photoImg1
						#show_photo_label_1.configure(image=photoImg1)
					
					#if img_counter-10 >= 0 :
						#img2 = Image.open("hinh/anh_%d.png"%(img_counter-10))
						#img2 = img2.resize((250,250), Image.ANTIALIAS)
						#photoImg2 =  ImageTk.PhotoImage(img2)
						#show_photo_label_2.image = photoImg2
						#show_photo_label_2.configure(image=photoImg2)
					
					print(img_counter,"------------------------")
					
					img = Image.open("./dataset/img/%s/anh_0.png"%result_names)
					img = img.resize((150,150), Image.ANTIALIAS)
					photoImg =  ImageTk.PhotoImage(img)
					show_photo_label.image = photoImg
					show_photo_label.configure(image=photoImg)
					if vvv!=result_names:
						with open("./dataset/train/"+result_names+"/"+result_names+".txt", "r") as x:
							image_1=Label(root, text=x.read(), font=("Arial",12) , bg="#F7F8FD",justify=LEFT)
							image_1.place(x=870,y=30)
							vvv=result_names

					if img_counter%2==0:
						sss=result_names
						if img_counter>=1 and sss!=xxx:
							img1 = Image.open("./dataset/img/%s/anh_0.png"%xxx)
							img1 = img1.resize((150,150), Image.ANTIALIAS)
							photoImg1 =  ImageTk.PhotoImage(img1)
							show_photo_label_1.image = photoImg1
							show_photo_label_1.configure(image=photoImg1)
							with open("./dataset/train/"+xxx+"/"+xxx+".txt", "r") as y:
								image_2=Label(root, text=y.read(), font=("Arial",12) , bg="#F7F8FD")
								image_2.place(x=870,y=200)
							if zzz is None:
								print("Chưa có zzz")
							else: #img_counter>=2:
								print("zzz:",zzz)
								if zzz!=0:
									img2 = Image.open("./dataset/img/%s/anh_0.png"%zzz)
									img2 = img2.resize((150,150), Image.ANTIALIAS)
									photoImg2 =  ImageTk.PhotoImage(img2)
									show_photo_label_2.image = photoImg2
									show_photo_label_2.configure(image=photoImg2)
									with open("./dataset/train/"+zzz+"/"+zzz+".txt", "r") as z:
										image_3=Label(root, text=z.read(), font=("Arial",12) , bg="#F7F8FD")
										image_3.place(x=870,y=370)
							zzz=xxx


					if img_counter%2!=0:
						xxx=result_names
						if xxx!=sss and img_counter>=1:
							img1 = Image.open("./dataset/img/%s/anh_0.png"%sss)
							img1 = img1.resize((150,150), Image.ANTIALIAS)
							photoImg1 =  ImageTk.PhotoImage(img1)
							show_photo_label_1.image = photoImg1
							show_photo_label_1.configure(image=photoImg1)
							with open("./dataset/train/"+sss+"/"+sss+".txt", "r") as y:
								image_2=Label(root, text=y.read(), font=("Arial",12) , bg="#F7F8FD")
								image_2.place(x=870,y=200)
							if zzz is None:
								print("Chưa có zzz")
							else: #img_counter>=1:
								print("zzz:",zzz)
								if zzz!=0:
									img2 = Image.open("./dataset/img/%s/anh_0.png"%zzz)
									img2 = img2.resize((150,150), Image.ANTIALIAS)
									photoImg2 =  ImageTk.PhotoImage(img2)
									show_photo_label_2.image = photoImg2
									show_photo_label_2.configure(image=photoImg2)
									with open("./dataset/train/"+zzz+"/"+zzz+".txt", "r") as z:
										image_3=Label(root, text=z.read(), font=("Arial",12) , bg="#F7F8FD")
										image_3.place(x=870,y=370)
							zzz=sss

					img_counter+=1

					#------------------------------------------
					resized = imutils.resize(frame, width=200)
					#plot result idx under box
					text_x = bb[k][0]
					text_y = bb[k][3] + 20
					#print('-----Percent ', best_class_probabilities[0])
					#print('-----Result Indices:  ', best_class_indices[0])
					#print('-----result_names ', result_names)
					
					cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
					text = "{}".format(result_names)
					cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=2, lineType=2)
					sss=result_names
				elif (best_class_probabilities > 0.05):
					result_names = 'unknown'
					#print('-----Percent ', best_class_probabilities[0])
					#print('-----Result Indices:  ', best_class_indices[0])
					#print('-----result_names ', result_names)

					text_x = bb[k][0]
					text_y = bb[k][3] + 20
					cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
					text = "{}".format(result_names)
					cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=2, lineType=2)
				k= k + 1

	#c = c + 1
	fps.update()
	#cv2.imshow('Video', frame)
	#---------------------------------------------------------------------------------------------------------
	#frame = cv2.flip(frame, 1)						# flip: hàm xoay ngược, lật ảnh , gồm có : -1, 0, 1  (link :https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/	)
	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)# Lọc ảnh từ BGR sang RBGA, ở dạng mảng
	img = Image.fromarray(cv2image)					# Image.fromarray: Tạo ra hình ảnh từ một đối tượng mảng (sử dụng giao thức đệm).
	imgtk = ImageTk.PhotoImage(image=img)			# ImageTk.PhotoImage: Load hình lên ('từ')
	show_video_label.image = imgtk					# dòng này có ý nghĩ là:	show_video_label = Label(image=imgtk)
	show_video_label.configure(image=imgtk)			# configure: hiển thị hình ảnh lên giao diện Tk // like set up background color, image
	show_video_label.place(x=40, y=30)				# đặt ở vị trí
	show_video_label.after(10, show_frame)			# sau 0.01 giây, show từng frame ảnh
	#----------------------------------------------------------------------------------------------------------
	#if cv2.waitKey(20) & 0xFF == ord('q'):
	#	break
vvv=0
zzz=0
img_counter = 0

root = Tk()
#root = Toplevel(formLogin)
root.title("Video Recognition tkiner")
root.geometry("1150x600+0+0")

img9 = PhotoImage(file="./logo face/landscape.png")
panel = Label(root, image = img9)
panel.pack(side = "bottom", fill = "both", expand = "yes")

btn_exit = Button(root, text="EXIT", font=("san-serif", 16, "bold"), fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", width=7, height=2, command=exit_click)
btn_exit.place(x = 360,y = 530)

show_video_label = Label(root)
show_video_label.pack()

labelframe1 = LabelFrame(root, width=380, height=170)  
labelframe1.place(x=690,y=20)

labelframe2 = LabelFrame(root, width=380, height=170)  
labelframe2.place(x=690,y=190)

labelframe3 = LabelFrame(root, width=380, height=170)  
labelframe3.place(x=690,y=360)

show_photo_label = Label(root, text="hình 1")
show_photo_label.place(x=700, y=30)

show_photo_label_1 = Label(root, text="hình 2")
show_photo_label_1.place(x=700, y=200)

show_photo_label_2 = Label(root, text="hình 3")
show_photo_label_2.place(x=700, y=370)

#-------------------------------------------------------------------------------------
path = "/home/tan/file_a_lan_gui/face_20_06/Image"

with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = dfa.create_mtcnn(sess, 'model/npy')

		minsize = 20  # minimum size of face                        # kích thước tối thiểu của khuôn mặt
		threshold = [0.6, 0.7, 0.7]  # three steps's threshold      # ngưỡng của ba bước    
		factor = 0.709  # three steps's threshold                   # ngưỡng của ba bước
		margin = 44
		frame_interval = 7
		batch_size = 1000
		image_size = 182
		input_image_size = 160
		
		print('------------ Loading Models-------------')
		names = pickle.loads(open("model/embeddings.pickle", "rb").read())
		abl.load_model('./model/face_model.pb')
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		embedding_size = embeddings.get_shape()[1]

		classifier_filename_exp = os.path.expanduser('model/model_recg.pkl')
		with open(classifier_filename_exp, 'rb') as infile:
			(model, class_names) = pickle.load(infile)

		#video_capture = VideoStream(src=0).start() 
		video_capture = cv2.VideoCapture(0)
		c = 0

		fps = FPS().start()
		print('Start Recognition')  #------> chạy tới đây
		prevTime = 0
		name_temp ='unknown'

#-------------------------------------------------------------------------------------

root.bind('<Escape>', lambda e: root.destroy())							# Nhấn Nút ESC Để phá cửa sổ

show_frame()
root.resizable(False, False)											# không cho điều chỉnh kích thước cửa sổ, kéo ra kéo vào
root.mainloop()														# Lưu ý :

