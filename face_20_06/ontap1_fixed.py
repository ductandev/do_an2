# Description: Label + Button + Entry + xử lý USER && PASSWORD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import pickle
from preprocess import preprocesses
from classifier import training

from tkinter import *
from tkinter import ttk 
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import shutil
import os

import tensorflow as tf
from scipy import misc
import numpy as np
import abl
import dfa
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import imutils


def btn_train_click():									
	#btn_train.place_forget()
	btn_train = Button(formLogin, text="Train Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", command=btn_train_click, state=DISABLED)
	btn_train.place(x = 210,y = 170)
	#file=exec(open("train.py").read())						#Run file python <name>.py 	*
	#----------------------------------------------------------------
	input_datadir = './dataset/train'
	output_datadir = "./dataset/img"
	modeldir = './model/face_model.pb'
	classifier_filename = './model/model_recg.pkl'
	names = os.listdir(input_datadir)
	names.sort()
	print ("Training Start")
	obj=preprocesses(input_datadir,output_datadir)
	nrof_images_total,nrof_successfully_aligned=obj.collect_data()

	print('----------- Total number of images: %d-----------' % nrof_images_total)
	print('----Number of successfully aligned images: %d---' % nrof_successfully_aligned)
	print ("--------------- Training ... ------------------")
	obj=training(output_datadir, modeldir, classifier_filename)

	get_file=obj.main_train()
	print('Saved model to file "%s"' % get_file)
	f = open("model/embeddings.pickle", "wb")
	f.write(pickle.dumps(names))
	#sys.exit("All Done")
	print("All Done")
	#----------------------------------------------------------------
	print("-------Train file success--------- ")
	btn_train = Button(formLogin, text="Train Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", command=btn_train_click, state=DISABLED)
	btn_train.place(x = 210,y = 170)
	#btn_train.place(x = 210,y = 170)


def btn_recogn_click():
	btn_recogn = Button(formLogin, text="Recognition", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_recogn_click, state=DISABLED)
	btn_recogn.place(x = 410,y = 170)
	#file=exec(open("tkinter_recognize.py").read())							#Run file python <name>.py 	*
	#---------------------------------------------------------------------------------------------------
	def exit_click():									# Cú pháp Button phải đặt dưới vòng def exit_btn() thì button mới nhảy vào trong được
		print("Clode Window slave success !")
		fps.stop()                                                                                                                              
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		video_capture.release()
		root.destroy()									# chỉ tắt cửa sổ con
		btn_recogn = Button(formLogin, text="Recognition", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_recogn_click)
		btn_recogn.place(x = 410,y = 170)


	def show_frame():
		global c
		global img_counter
		global sss
		global xxx
		global zzz
		global vvv
		
		if (root.destroy):
			btn_recogn = Button(formLogin, text="Recognition", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_recogn_click)
			btn_recogn.place(x = 410,y = 170)
		
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
					print(predictions)
					best_class_indices = np.argmax(predictions, axis=1)
					best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
					# print("predictions")
					#print(best_class_indices,' with accuracy ',best_class_probabilities)

					# print(best_class_probabilities)
					result_names = names[best_class_indices[0]]
					if (best_class_probabilities >= 0.15) and (result_names != 'unknown'):
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
						#print(img_counter,"------------------------")

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
									image_2=Label(root, text=y.read(), font=("Arial",12) , bg="#F7F8FD",justify=LEFT)
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
											image_3=Label(root, text=z.read(), font=("Arial",12) , bg="#F7F8FD",justify=LEFT)
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
									image_2=Label(root, text=y.read(), font=("Arial",12) , bg="#F7F8FD",justify=LEFT)
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
											image_3=Label(root, text=z.read(), font=("Arial",12) , bg="#F7F8FD",justify=LEFT)
											image_3.place(x=870,y=370)
								zzz=sss

						img_counter+=1
						#------------------------------------------
						resized = imutils.resize(frame, width=200)
						#plot result idx under box
						text_x = bb[k][0]
						text_y = bb[k][3] + 20
						print('-----Percent ', best_class_probabilities[0])
						print('-----Result Indices:  ', best_class_indices[0])
						print('-----result_names ', result_names)
						
						cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
						text = "{}".format(result_names)
						cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=2, lineType=2)
					elif (best_class_probabilities > 0.05):
						result_names = 'unknown'
						print('-----Percent ', best_class_probabilities[0])
						print('-----Result Indices:  ', best_class_indices[0])
						print('-----result_names ', result_names)

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

	
	#root = Tk()
	root = Toplevel(formLogin)
	root.title("Video Recognition tkiner")
	root.geometry("1150x600+0+0")
	#root.configure(bg='#F7F8FD')										# set up background color

	panel = Label(root, image = background_image)						# show background on Label root
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
	path = "/home/tan/face_20_06/Image"

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

			fps = FPS().start()
			print('Start Recognition')  #------> chạy tới đây
			prevTime = 0
			name_temp ='unknown'

	#-------------------------------------------------------------------------------------
	root.bind('<Escape>', lambda e: root.destroy())							# Nhấn Nút ESC Để phá cửa sổ
	root.resizable(False, False)											# không cho điều chỉnh kích thước cửa sổ, kéo ra kéo vào
	show_frame()
	#---------------------------------------------------------------------------------------------------

def btn_addface_click():
	list = lg1, btn_addface, btn_train, btn_recogn							# Quên lg1 đi có thể hiện lại bằng lệnh: lg1.place(x=120,y=145)
	for i in list:									
		i.place_forget()
	#------------------------------------------------------------------------------------------------------------------
	def btnLogin_click():
		user = tbUser.get()
		passwd = tbPass.get()
		#if (user=="admin" & passwd=="123"):								# trong Python không thể so sánh 2 chuỗi str nên phải tách ra xử lý
		if (user =="admin"):
			if(passwd =="123"):
				tbUser.delete(0, END)										#xóa hiển thị đã nhập vào user
				tbPass.delete(0, END)										#xóa hiển thị đã nhập vào pass
				
				messagebox.showinfo("THÔNG BÁO", "Đăng nhập thành công.")
				dllb1 = Label(formLogin, text = "Đăng nhập thành công", fg = "green" ,font = ("calibri", 11))	#dllb : delete Label1
				#dllb1.pack()
				#dllb1.after(1000 , dllb1.destroy)							#xóa Label sau 1 giây
				
				list1 = lg0,lg1,lb2,lb3,tbUser,tbPass,btnLogin	
				for i in list1:									
					i.place_forget()										# xóa lg0,lg1,lb1,lb2,lb3,tbUser,tbPass,btnLogin
					#i.pack_forget()

				formLogin.geometry("600x400+400+100")
				
				lg1.place(x=235,y=20)										# đặt logo1
				
				btn_addface = Button(formLogin, text="Add Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_addface_click,state=DISABLED)
				btn_addface.place(x = 10,y = 170)
				btn_train.place(x = 210,y = 170)
				btn_recogn.place(x = 410,y = 170)
				
				formLogin.bind('<Escape>', lambda e: formLogin.quit())		# Nhấn Nút ESC Để Thoát
				#------------------------------------------------------------------------------------------------------------
				
				#file=exec(open("xx.py").read())							#Run file python <name>.py 	*
				#print("Run file 1 success ")

				def btn_delete_click():
					deletedir = filedialog.askdirectory(initialdir = "/home/tan/face_20_06/dataset/train/", title='Chọn thư mục để xóa') #deletedir: delete directory
					print(deletedir)
					if not deletedir:										# Nếu ko có deletedir
						print("Closed")
					else:
						#os.chdir("/home/tan/AI/giao dien Tkinter/test")	# thay đổi thư mục làm việc hiện tại thành đường dẫn đã chỉ định
						#now = os.getcwd()									
						#os.remove(): chỉ xóa được file, không thể xóa hoặc xóa một thư mục
						#os.rmdir() : được sử dụng để xóa hoặc xóa một thư mục trống
						shutil.rmtree(deletedir)# được sử dụng để xóa toàn bộ cây thư mục
						print("Delete folder ----- %s ----- successed\n"%deletedir)

				def btn_clear_click():
					tbname.delete(0,END)
					tb_birthday.delete(0,END)
					tb_phone.delete(0,END)
					combo.current(0)

				def btn_exit_click():
					win.destroy()
					#btn_addface.place(x = 10,y = 170)
					btn_addface = Button(formLogin, text="Add Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_addface_click)
					btn_addface.place(x = 10,y = 170)

				def btn_ok_click():
					name = tbname.get()
					birthday = tb_birthday.get()
					phone = tb_phone.get()
					dep = entry7.get()
					
					path = "/home/tan/face_20_06/dataset/train/"+name
					if not name:												# Nếu ko có name
						messagebox.showerror("THÔNG BÁO", "Vui lòng nhập tên")
					elif name.find("  ")!= -1 or name==" " or name.find(" ")==0:
						messagebox.showerror("THÔNG BÁO", "Sai cú pháp vui lòng nhập lại !")
					elif os.path.exists(path):
						messagebox.showerror("THÔNG BÁO", "Tên đã tồn tại vui lòng chọn tên khác !")
					elif not birthday:
						messagebox.showerror("THÔNG BÁO", "Vui lòng nhập ngày sinh !")
					elif not phone:
						messagebox.showerror("THÔNG BÁO", "Vui lòng nhập số điện thoại !")
					elif not dep and dep=="":
						messagebox.showerror("THÔNG BÁO", "Vui lòng nhập chức vụ !")
					else:	
						#os.makedirs											# os.makedirs: tạo được nhiều thư mục 1 lúc như tạo thư mục mẹ rồi tạo tiếp thư mục con trong đó và nhiều hơn được nữa.
						os.mkdir(path)											# os.mkdir:  mặc định chỉ tạo được 1 thư mục 
						create_file=open(path+"/"+ name + ".txt", "w")
						create_file.write("Tên: "+ name + "\n")
						create_file.write("Ngày sinh: "+ birthday +"\n")
						create_file.write("Số điện thoại: "+ phone +"\n")
						create_file.write("Chức vụ: "+ dep)
						create_file.close()
						print("-------Running face data creation -------")
						file1 = exec(open("./dataset.py").read())
						print("Done")
						print("-----------------------------")
						btn_ok.place_forget()
						
						dllb5 = Label(win, text = "Đăng ký thành công", fg = "green" ,font = ("calibri", 12), bg="#F7F8FD")	#dllb : delete Label5
						dllb5.place(x=100,y=160)
						dllb5.after(2000 , dllb5.destroy)
						
						tbname.delete(0,END)
						tb_birthday.delete(0,END)
						tb_phone.delete(0,END)
						combo.current(0)
						
						btn_ok.place(x=252,y=200)
				
				win = Toplevel(formLogin)
				win.geometry("340x250+450+170")
				win.title("registration form")
				win.configure(bg='#F7F8FD')										#set up background color

				lb4 = Label(win, text="Tên:", font=("Arial",12), bg="#F7F8FD")
				lb4.place(x=10,y=20)
				tbname = Entry(win, width=18, font=("Consolas", 12)) 			#tbname : Textbox_name
				tbname.place(x=120,y=20)

				lb5 = Label(win, text="Ngày sinh:", font=("Arial",12), bg="#F7F8FD")
				lb5.place(x=10,y=55)
				tb_birthday = Entry(win, width=18, font=("Consolas", 12)) 		#tbname : Textbox_name
				tb_birthday.place(x=120,y=55)

				lb6 = Label(win, text="Số điện thoại:", font=("Arial",12), bg="#F7F8FD")
				lb6.place(x=10,y=90)
				tb_phone = Entry(win, width=18, font=("Consolas", 12)) 			#tbname : Textbox_name
				tb_phone.place(x=120,y=90)

				lb7 = Label(win, text="Chức vụ:", font=("Arial",12), bg="#F7F8FD")
				lb7.place(x=10,y=125)
				entry7=StringVar()
				combo = ttk.Combobox(win, textvariable=entry7, width=12, font=("san-serif", 11, "bold"), state='readonly')
				combo['values']=("","Giáo viên","Giám đốc","Trưởng Phòng","Phó Phòng","Quản lý","Nhân viên","Sinh viên")
				combo.place(x=120,y=125)

				btn_delete = Button(win, text="Delete", font=("san-serif", 16, "bold"), width = 4, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", command=btn_delete_click)
				btn_delete.place(x=5,y=200)
					
				btn_clear = Button(win, text="Clear", font=("san-serif", 16, "bold"), width = 3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_clear_click)
				btn_clear.place(x=97,y=200)

				btn_exit = Button(win, text="Exit", font=("san-serif", 16, "bold"), width = 3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_exit_click)
				btn_exit.place(x=174,y=200)

				btn_ok = Button(win, text="OK", font=("san-serif", 16, "bold"), width = 3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_ok_click)
				btn_ok.place(x=252,y=200)

				win.bind('<Escape>', lambda e: win.destroy())
				win.resizable(False, False)									# không cho điều chỉnh kích thước cửa sổ, kéo ra kéo vào 

				#------------------------------------------------------------------------------------------------------------
				
			else:
				messagebox.showerror("THÔNG BÁO", "Đăng nhập thất bại.")
				dllb2 =	Label(formLogin, text = "Đăng nhập thất bại", fg = "red", bg="#F7F8FD" ,font = ("calibri", 11)) 	#dllb : delete Label2
				dllb2.place(x=121,y=470)
				dllb2.after(1000 , dllb2.destroy)							#xóa Label sau 1 giây
				
		else:
			messagebox.showerror("THÔNG BÁO", "Đăng nhập thất bại.")
			dllb3 = Label(formLogin, text = "Đăng nhập thất bại", fg = "red", bg="#F7F8FD" , font = ("calibri", 11))		#dllb : delete Label3
			dllb3.place(x=121,y=470)
			dllb3.after(1000 , dllb3.destroy)								#xóa Label sau 1 giây
	#--------------------------------------------------------------------------------------------------------------------

	formLogin.geometry("360x565+500+0") 
	formLogin.title("Form đăng nhập ")
	#formLogin.configure(bg='#F7F8FD')										#set up background color

	lg0.place(x=25,y=5) 
	
	lg1.place(x=120,y=145)													# Đặt vị trí

	#lb1 = Label(formLogin, text="Adminstator login form", font=("Times New Roman", 22,"bold","italic"), fg="red", bg="#F7F8FD")
	#lb1.place(x=40,y=240)

	lb2 = Label(formLogin, text="Tên đăng nhập ", font=("Arial",12), bg="#F7F8FD")
	lb2.place(x=30,y=280)

	#Entry: lệnh dùng để nhập vào
	tbUser = Entry(formLogin, width=30, font=("Consolas", 12)) 				#tbUser : Textbox_user
	tbUser.place(x=30,y=305)

	lb3 = Label(formLogin, text="Mật Khẩu ", font=("Arial",12), bg="#F7F8FD")
	lb3.place(x=30,y=335) 

	#Entry: lệnh dùng để nhập vào
	tbPass = Entry(formLogin, width=30, font=("Consolas", 12), show="*")	#tbPass : Textbox_Pass
	tbPass.place(x=30,y=360)

	btnLogin = Button(formLogin, text="Đăng nhập", font=("san-serif", 16, "bold"), height = 1, width = 19, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", command=btnLogin_click)
	btnLogin.place(x=28,y=420)

	formLogin.bind('<Escape>', lambda e: formLogin.destroy())


img_counter = 0
c = 0
zzz=0
vvv=0

formLogin = Tk()
formLogin.geometry("600x405+400+100")
formLogin.title("Face recognition interface")
formLogin.configure(bg='#F7F8FD')											#set up background color

background_image = PhotoImage(file="./logo face/landscape.png")				#set up background root tab		

logo = Image.open("/home/tan/face_20_06/logo face/iuh.png")
photo = ImageTk.PhotoImage(logo)
lg0 = Label(image=photo,bg="#F7F8FD")										#lg0: logo 0

logo1 = Image.open("/home/tan/face_20_06/logo face/logo9.png")
photo1 = ImageTk.PhotoImage(logo1)											# Load hình lên
lg1 = Label(image=photo1,bg="#F7F8FD")										# Truyền lên Label 
lg1.place(x=235,y=20)														# Đặt vị trí

menubar = Menu(formLogin)													#lệnh gọi sử dụng Menu
file = Menu(menubar, tearoff=0)												# tearoff=0 : bắt đầu từ ví trí thứ 0 trong Menu
file.add_command(label="New")   
file.add_separator()														# dấu gặch ngăn cách [________]
file.add_command(label="Exit", command=formLogin.quit)  
menubar.add_cascade(label="File", menu=file)								# add_cascade : lệnh thêm vào list "File"

about = Menu(menubar, tearoff=0)
def about_click():
	messagebox.showinfo("About license", "This interface make by:\n + Nguyễn Đức Tấn 16026631\n + Nguyễn Thế Hiển 16031901  ")
about.add_command(label="About interface face recognition", command=about_click)  
menubar.add_cascade(label="About", menu=about)

menubar.add_command(label="Quit!", command=formLogin.quit)
# display the menu 
formLogin.config(menu=menubar)

btn_addface = Button(formLogin, text="Add Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_addface_click)
btn_addface.place(x = 10,y = 170)
btn_train = Button(formLogin, text="Train Face", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange", command=btn_train_click)
btn_train.place(x = 210,y = 170) 
btn_recogn = Button(formLogin, text="Recognition", font=("san-serif", 16, "bold"),width=10, height=3, fg="white", bg="#1380C3",activeforeground = "white",activebackground = "orange",command=btn_recogn_click)
btn_recogn.place(x = 410,y = 170)

formLogin.bind('<Escape>', lambda e: formLogin.quit())						# Nhấn Nút ESC Thoát
formLogin.resizable(False, False)											# không cho điều chỉnh kích thước cửa sổ, kéo ra kéo vào
formLogin.mainloop()


