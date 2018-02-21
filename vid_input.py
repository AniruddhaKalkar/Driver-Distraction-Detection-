import tflearn
import os
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import pickle
from random import shuffle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
'''
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture('lana.mp4')
print('vid found')
while(cap.isOpened()):
    print('inside')
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Face",frame)
    if(cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
'''
IMG_SIZE=64
LR=1e-3
MODEL_NAME = 'driverdistraction-{}-{}.model'.format(LR, '2conv-basic')

classes=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']

values = np.array(classes)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

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

cla=[]
cap=cv2.VideoCapture('lana.mp4')
#print('vid found')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        #print('Loading Model...')
        while(cap.isOpened()):
            #print('inside')
            ret,frame=cap.read()
            frame=cv2.resize(frame,(64,64))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            features=np.array(gray)
            #print(features.shape)
            features=np.array(features).reshape(64,64,1)
            #print(features.shape)
            pred=model.predict([features])
            #pred=pred.reshape(10,1)
            #print(pred)
            lbl=label_encoder.inverse_transform([np.argmax(pred)])
            cv2.imshow('frame',frame)
            cla.append(lbl)
        
            print(lbl)
            if(cv2.waitKey(1)==ord('q')):
                break
print('i')
print(np.array(cla).shape)
cap.release()
cv2.destroyAllWindows()
