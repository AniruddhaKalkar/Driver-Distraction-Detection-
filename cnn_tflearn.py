import tflearn
import os
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import pickle
from random import shuffle
IMG_SIZE=64
LR=1e-3
MODEL_NAME = 'driverdistraction-{}-{}.model'.format(LR, '2conv-basic')

with open('Training_features.pickle','rb') as f:
    train_data=pickle.load(f)

shuffle(train_data)
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

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Loading Model...') 

train=train_data[:-1000]
test=train_data[-1000:]
X=np.array([i[0]for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in test]

model.fit(X,Y,n_epoch=10,validation_set=(test_x,test_y),snapshot_step=500,show_metric=True,run_id=MODEL_NAME)

model.save(MODEL_NAME)
