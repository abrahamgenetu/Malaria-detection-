import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np 
import pandas as pd
import os


#change the path to the directory where images are located
path="C:\\Users\\mihir\\Desktop\\IST 707\\datasets\\cell_images"

#image processing 
img=[]
labels=[]

Parasitized=os.listdir(path+"/Parasitized/")
for a in Parasitized:
    try:
        image=cv2.imread(path+"/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        img.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")
		

Uninfected=os.listdir(path+"/Uninfected/")
for b in Uninfected:
    try:
        image=cv2.imread(path+"/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        img.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")
		
		
img=np.array(img)
labels=np.array(labels)

len_data=len(img)

s=np.arange(img.shape[0])
np.random.shuffle(s)
img=img[s]
labels=labels[s]

#model building
def get_model(x_train,y_train):
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(250,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation="softmax"))#2 represents output layer neurons 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=50,epochs=30,verbose=1,validation_split=0.1)
    return model




#model evaluation
def evaluate_model(model,x_test,y_test):
    metric_dict={}
    pred=model.predict(x_test)
    pred=np.argmax(pred,axis=1)
    pred=np_utils.to_categorical(pred)
    metric_dict["acc"]=accuracy_score(y_test, pred)
    metric_dict["recall"]=recall_score(y_test,pred,average='weighted')
    metric_dict["auc"]=roc_auc_score(y_test,pred,average='weighted')
    return metric_dict
	
	
#3-fold acc,auc,recall
kFold = StratifiedKFold(n_splits=3)
eval_list=list()
for train, test in kFold.split(img, labels):
    x_train=img[train]
    x_train = x_train.astype('float32')/255 
    x_test=img[test]
    x_test = x_test.astype('float32')/255 
    y_train=labels[train]
    y_train=np_utils.to_categorical(y_train)
    y_test=labels[test]
    y_test=np_utils.to_categorical(y_test)
    
    model=get_model(x_train,y_train)
    met_dict=evaluate_model(model,x_test,y_test)
    eval_list.append(met_dict)

Accuracy=0.
AUC=0.
Recall=0.
for d in eval_list:
    Accuracy+=d["acc"]
    AUC+=d['auc']
    Recall+=d["recall"]
Accuracy=Accuracy/3
AUC=AUC/3 
Recall=Recall/3

print("3-fold CV accuracy is {:.10f}".format(Accuracy))
print("3-fold CV AUC is {:.10f}".format(AUC))
print("3-fold CV Recall is {:.10f}".format(Recall))