import numpy as np
import csv
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
import pickle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE=64
LR=1e-3
MODEL_NAME = 'driverdistraction-{}-{}.model'.format(LR, '2conv-basic')
convnet=tflearn.layers.core.input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet=conv_2d(convnet,32,5,activation='relu')
#convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,128,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)


convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,10,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(convnet,tensorboard_dir='log')
count=1
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Loading Model...') 
    with open('submission1.csv','w') as f:
        f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
        for i in range(1,9):
            p='Test_data'+str(i)+'.npy'
            data=np.load(p)
            print(data.shape)
            for tup in data:
                img_id=tup[1]
                features=np.array(tup[0])
                #print(features.shape)
                features=np.array(features).reshape(64,64,1)
                #print(features.shape)
                pred=model.predict([features])
                pred=pred.reshape(10,1)
                #print(img_id)
                #print(pred)
                print(count)
                count=count+1
                f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(img_id,float(pred[0]),float(pred[1]),float(pred[2]),float(pred[3]),float(pred[4]),float(pred[5]),float(pred[6]),float(pred[7]),float(pred[8]),float(pred[9])))
print('i')        
        
