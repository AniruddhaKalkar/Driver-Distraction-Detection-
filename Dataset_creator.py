import numpy as np
import csv
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
import pickle 
Train_Dir='C:\\Distraction\\train'
Test_Dir='C:\\Distraction\\test'

classes=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']

values = np.array(classes)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

def getLabel(i):
    print(i)
    lbl=onehot_encoder.transform(label_encoder.transform([str(i)]))
    return lbl
    
def reshape_train():
    train_data=[]
    for classes in os.listdir(Train_Dir):
        p=os.path.join(Train_Dir,classes)
        label=getLabel(classes)
        for im in os.listdir(p):
            path=os.path.join(p,im)
            image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            image=cv2.resize(image,(64,64))
            features=list(np.array(image))
            label=np.array(label).reshape(10,)
            label=list(label)
            train_data.append([features,label])
            scipy.misc.imsave('Train_data/'+im,image)
    with open('Training_features.pickle','wb') as f:
        pickle.dump(train_data,f)
    return train_data

train_data=reshape_train()
