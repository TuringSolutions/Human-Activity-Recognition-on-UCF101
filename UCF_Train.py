# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:27:58 2020
@author: danish
Dependencies:
    conda install -c conda-forge opencv
    pip install scikit-image
    pip install tqdm 
"""

from VideoPreprocessing import VideoNameDF, TagVideos, Video2Frames, FramesCSV
from UCF_Model import ReadFrames, DatasetSplit, VGG16Model, UCFModel_Train


###################### Video Preprocessing ########################
# creating a dataframe having video names
train = VideoNameDF(name='trainlist01.txt', dir='ucfTrainTestlist')
test =  VideoNameDF(name='testlist01.txt', dir='ucfTrainTestlist')

#Next, we will add the tag of each video (for both training and test sets).
train = TagVideos(train)
test = TagVideos(test)

#Extracting frames from training videos.
path = 'D:/Study/DataScience/DataSets/UCF'
status = Video2Frames(train, frames_dir=path+'/train_1', videos_dir=path+'/UCF-101')
print(status)

#Save the names of the frames to a CSV file along with their corresponding tags.
train_data = FramesCSV(frames_dir=path+'/train_1', csv_dir='UCF', csv_name='train_new.csv')

################### Data Preprocessing ##########################    

import pandas as pd
train = pd.read_csv('UCF/train_new.csv')
train.head()

X = ReadFrames(train, frames_dir=path+'/train_1')
#Creating the test set and validation set.
X_train, X_test, y_train, y_test = DatasetSplit(train, X)
#delete the variable X to save the space.
del X
################### Train the Model ##########################
X_train, X_test = VGG16Model(X_train, X_test)
history = UCFModel_Train(X_train, y_train, X_test, y_test, epochs=200, ckpt_name='UCF_weights.h5')



