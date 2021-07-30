from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle
from preprocess import preprocesses
from classifier import training

input_datadir = './dataset/train'
output_datadir = "./dataset/img"
modeldir = './model/face_model.pb'
classifier_filename = './model/model_recg.pkl'
names = os.listdir(input_datadir)
print(names)
print("----------------------------------------------")
names.sort()
print(names)

#print ("Training Start")
#obj=preprocesses(input_datadir,output_datadir)
#nrof_images_total,nrof_successfully_aligned=obj.collect_data()

#print('----------- Total number of images: %d-----------' % nrof_images_total)
#print('----Number of successfully aligned images: %d---' % nrof_successfully_aligned)
#print ("--------------- Training ... ------------------")
#obj=training(output_datadir, modeldir, classifier_filename)

#get_file=obj.main_train()
#print('Saved model to file "%s"' % get_file)
#f = open("model/embeddings.pickle", "wb")
#f.write(pickle.dumps(names))
#sys.exit("All Done")

