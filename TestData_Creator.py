import numpy as np
import csv
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
import pickle 
Test_Dir='C:\\Distraction\\test'
    
def reshape_test():
    test_data=[]
    c=1
    for img in os.listdir(Test_Dir):
            path=os.path.join(Test_Dir,img)
            image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            image=cv2.resize(image,(64,64))
            img_nm=str(img)
            print(c)
            c=c+1
            features=list(np.array(image))
            test_data.append([features,img_nm])
            scipy.misc.imsave('Test_data/'+img,image)
    print('all imgs saved')
    test_data1=test_data[:10000]
    np.save('Test_data1.npy',test_data1)
    print('1st.npy created!')  
    test_data2=test_data[10000:20000]
    np.save('Test_data2.npy',test_data2)
    print('1st.npy created!')
    test_data3=test_data[20000:30000]
    np.save('Test_data3.npy',test_data3)
    print('1st.npy created!')
    test_data4=test_data[30000:40000]
    np.save('Test_data4.npy',test_data4)
    print('1st.npy created!')
    test_data5=test_data[40000:50000]
    np.save('Test_data5.npy',test_data5)
    print('1st.npy created!')
    test_data6=test_data[50000:60000]
    np.save('Test_data6.npy',test_data6)
    print('1st.npy created!')
    test_data7=test_data[60000:70000]
    np.save('Test_data7.npy',test_data7)
    print('1st.npy created!')
    test_data8=test_data[70000:]
    np.save('Test_data8.npy',test_data8)
    print('1st.npy created!')
    
reshape_test()
