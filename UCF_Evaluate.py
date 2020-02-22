# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:17:22 2020

@author: danish
"""

import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from UCF_Model import UCFModel, PredictAction, PredictOnBatch
from VideoPreprocessing import VideoNameDF
from sklearn.metrics import accuracy_score


path = 'D:/Study/DataScience/DataSets/UCF'

base_model = VGG16(weights='imagenet', include_top=False)
model = UCFModel(shape=25088)
# loading the trained weights
model.load_weights(path+'/ckpt/UCF_weights.hdf5')
# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# getting the test list
test =  VideoNameDF(name='testlist01.txt', dir='ucfTrainTestlist')
test_videos = test['video_name']

# creating the tags
train = pd.read_csv('UCF/train_new.csv')
y = train['class']
y = pd.get_dummies(y)


################# Making a prediction on single video ################
file='UCF/v_Archery_g03_c03.avi'
action = PredictAction(file, y, model, base_model)
print(action)
    
################# Making a prediction on batch of videos ################
predict, actual = PredictOnBatch(test_videos, y, base_model, model, videos_dir=path+'/UCF-101')
# checking the accuracy of the predicted tags
acc = accuracy_score(predict, actual)*100
print(acc)