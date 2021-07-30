from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import abl
import dfa
import os
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
import time
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import Jetson.GPIO as GPIO
import imutils
import csv
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText


emailfrom = "testopenhablan@gmail.com"
emailto = "lannlb1385@gmail.com"
fileToSend = "tss.csv"

msg = MIMEMultipart()
msg["From"] = emailfrom
msg["To"] = ",".join(emailto)
msg["Subject"] = "Send CSV"
msg.preamble = "SEND CSV2"
ctype, encoding = mimetypes.guess_type(fileToSend)
if ctype is None or encoding is not None:
    ctype = "application/octet-stream"
maintype, subtype = ctype.split("/", 1)
path = "/home/lan/Desktop/face_20_06/Image"
DOOR_PIN = 37
BUTT_PIN = 19
GPIO.setmode(GPIO.BOARD)
GPIO.setup(DOOR_PIN, GPIO.OUT)
GPIO.setup(BUTT_PIN, GPIO.IN)
GPIO.output(DOOR_PIN, GPIO.HIGH)


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = dfa.create_mtcnn(sess, 'model/npy')

        minsize = 20  
        threshold = [0.6, 0.7, 0.7]  
        factor = 0.709 
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
        video_capture = VideoStream(src='rtsp://192.168.1.100:554/h264&basic_auth=YWRtaW46MTIzNA==').start() 

        c = 0
        try:
            connection = mysql.connector.connect(host='localhost', database='access_db', user='root', password='lanluan1')
            print ("mysql is connected")
        except mysql.connector.Error as error :
            connection.rollback() #rollback if any exception occured
            print("Failed inserting record into db_face table {}".format(error))
        fps = FPS().start()
        print('Start Recognition')
        prevTime = 0
        alarm = False
        time_open_door = 0
        temp = 0
        name_temp ='unknown'
        no_frame = 0
        while True:
            frame = video_capture.read()

            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            if frame is None:
                no_frame = no_frame + 1
                if(no_frame >= 2):
                    time_er = datetime.datetime.now()
                    cursor = connection.cursor()
                    sql_insert_query = "INSERT INTO `camera_err` (`id`, `error_code`, `time_err`) VALUES (%s, %s,  %s)"
        
                    error_code="err_stream"
                    insert_tuple = (0, error_code, time_er)                                 
                    result  = cursor.execute(sql_insert_query, insert_tuple)
                    connection.commit()
                    print ("Record inserted successfully into camera_err table")
                    no_frame = 0
                    video_capture.stop()
                    fps.stop()
                    video_capture = VideoStream(src='rtsp://192.168.1.100:554/h264&basic_auth=YWRtaW46MTIzNA==').start() 
                    fps = FPS().start()
                time.sleep(10) 
                continue
            curTime = time.time()+1    # calc fps
            timeF = frame_interval
            if(alarm == False):
                if (c % timeF == 0): 
                    c = 0
                    find_results = []

                    if 0:
                        now = datetime.datetime.now()
                        time_begin = now.replace(hour=16, minute=27, second=0, microsecond=0)
                        time_end = now.replace(hour=16, minute=27, second=5, microsecond=0)
                        if((now > time_begin) and (now < time_end)):
                            print("send CSV file to report \n")
                            sql_select_Query = "select p.id, p.name, p.department, p.part, p.email, p.phone_number, p.image_path, p.birth_day , a.access_time from person_info p, access_detail a where p.name = a.name and a.access_time between %s and %s" 
                            cursor = connection.cursor()
                            today8am = now.replace(hour=6, minute=0, second=0, microsecond=0)
                            today9am = now.replace(hour=17, minute=30, second=0, microsecond=0)
                            insert_time = (today8am,today9am)
                            cursor.execute(sql_select_Query,insert_time)
                            result = cursor.fetchall()
                            with open('tss.csv', 'w') as writeFile:
                                csv_wr = csv.writer(writeFile)
                                for x in result:
                                    csv_wr.writerow(x)
                            writeFile.close()
                            fp = open(fileToSend)
                            # Note: we should handle calculating the charset
                            attachment = MIMEText(fp.read(), _subtype=subtype)
                            fp.close()
                            attachment.add_header("Content-Disposition", "attachment", filename=fileToSend)
                            msg.attach(attachment)
                            s = smtplib.SMTP('smtp.gmail.com', 587) 
    
                            # start TLS for security 
                            s.starttls() 
                            
                            # Authentication 
                            s.login(emailfrom, "taolaobidao1") 

                            s.sendmail(emailfrom, emailto, msg.as_string())
                            s.quit()


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
                            if (best_class_probabilities >= 0.16) and (result_names != 'unknown'):
                                cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
                                time_open_door = 0
                                percent = best_class_probabilities[0] * 100 + 40
                                if(percent > 100):
                                    percent = 100
                                if(alarm == False) :
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
                                    GPIO.output(DOOR_PIN, GPIO.LOW)
                                    datetime_object = datetime.datetime.now()
                                    cursor = connection.cursor(prepared=True)
                                    #sql_insert_query = """ INSERT INTO `access_detail` (`id`, `name`, `access_time`) VALUES (%s, %s,  %s)"""
                                    sql_insert_query = "INSERT INTO `access_detail` (`id`, `name`, `access_time`, `accuracy`, `name_2`, `accuracy_2`, `ratio`) VALUES (%s, %s,  %s, %s, %s, %s, %s)"
                                    name_2 = names[flag]
                                    insert_tuple = (0, result_names, datetime_object,percent,name_2,max2, max/max2 )                                   
                                    result  = cursor.execute(sql_insert_query, insert_tuple)
                                    lastid = cursor.lastrowid
                                    
                                    connection.commit()
                                    #print ("Record inserted successfully into person_login table")
                                    alarm = True
                                    time_open_door = 0
                                    resized = imutils.resize(frame, width=200)
                                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                                    p = os.path.sep.join([path +"/"+ result_names, "{}_{}.jpg".format(result_names, lastid)])
                                    cv2.imwrite(p, gray)
                                    #print("Open The Door")
                                #plot result idx under box
                                text_x = bb[k][0]
                                text_y = bb[k][3] + 20
                                #print('Result Indices: ', best_class_indices[0])
                                cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
                                text = "{}".format(result_names)
                                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=2, lineType=2)
                            elif (best_class_probabilities > 0.05):
                                result_names = 'unknown'
                                cv2.rectangle(frame, (bb[k][0], bb[k][1]), (bb[k][2], bb[k][3]), (0, 255, 0), 2)    #boxing face
                            k= k + 1
                            #cv2.imshow('Video', frame)
                
                c = c + 1
            if(alarm == True):
                time_open_door = time_open_door + 1
                if(time_open_door >= 350):
                    GPIO.output(DOOR_PIN, GPIO.HIGH)
                    time_open_door = 0
                    alarm = False
                    print("Close The Door")

            fps.update()
            cv2.imshow('Video', frame)
            if(GPIO.input(BUTT_PIN) == GPIO.LOW):
                temp = 1
                time_open_door_butt = 0
                GPIO.output(DOOR_PIN, GPIO.LOW)
            if(temp == 1):
                time_open_door_butt = time_open_door_butt + 1
                if(time_open_door_butt >= 100):
                    GPIO.output(DOOR_PIN, GPIO.HIGH)
                    time_open_door_butt = 0
                    temp = 0
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        fps.stop()
        if(connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        video_capture.stop()
        cv2.destroyAllWindows()
