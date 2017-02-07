import numpy as np
#import keras
import csv
import cv2
import os
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator

from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling


from random import randint
import tensorflow as tf
import json

iShowDebugPic =0


def generate_next_batch(batch_size=16):
    """
    This generator yields the next training batch
    :param batch_size:
        Number of training images in a single batch
    :return:
        A tuple of features and steering angles as two numpy arrays
    """
    while True:
        #X_batch = []
        #y_batch = []
        iIndex = randint(0,len(CenterIMGPath)-batch_size)
        
        #X_batch = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        X_batch = np.zeros([batch_size,64,2*64,ImgShape[2]])
        #LeftImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        #RightImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        y_batch = np.zeros(batch_size)

        for i in range(iIndex,iIndex+batch_size):
            iSelect = randint(0,2)
            if (iSelect==0):
                tmpImg = cv2.imread(CenterIMGPath[i],1)
                
                #X_batch[i-iIndex] =cv2.imread(CenterIMGPath[i],1)
                
            elif (iSelect==1):
                tmpImg =cv2.imread(LeftIMGPath[i].strip(),1)
                y_batch[i-iIndex] = SWA_hist[i]-0.2
            elif (iSelect==2):
                tmpImg =cv2.imread(RightIMGPath[i].strip(),1)
                y_batch[i-iIndex] = SWA_hist[i]+0.2
            

            if (iShowDebugPic==2):
                #plt.subplot(231)   
                plt.imshow(tmpImg)
                plt.show()
                #plt.subplot(232)   
                plt.imshow(cv2.resize(tmpImg,(2*64, 64), interpolation = cv2.INTER_CUBIC))
                plt.show()
                #plt.subplot(233)   
                plt.imshow(cv2.resize(tmpImg,(64, 2*64), interpolation = cv2.INTER_CUBIC))
                plt.show()

            X_batch[i-iIndex] = cv2.resize(tmpImg,(2*64, 64), interpolation = cv2.INTER_CUBIC)
            y_batch[i-iIndex] = SWA_hist[i]

        #X_batch = CenterImg
        #y_batch = SWA_corrected
        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Model save
    """

    json_string = model.to_json()
    print(json_string)
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


#sFilePathInput = r'C:\Users\Christoph\Documents\udacity\11_Cloning\simulator-windows-64\driving_log.csv'
sFilePathInput = r'driving_log.csv'
sPathReplace = 'IMG/'
#sPathReplace = 'C:\\Users\\Christoph\\Documents\\udacity\\11_Cloning\\simulator-windows-64\\IMG\\'
SWA_hist=[]
CenterIMGPath = []
LeftIMGPath = []
RightIMGPath = []
File = open(sFilePathInput,'r')
ImgShape =[]
csvInput = csv.reader(File, delimiter=',')
for row in csvInput:
    #print(row)
    #image = cv2.imread(row[0],0)
    SWA_hist.append(float(row[3]))
    CenterIMGPath.append(row[0].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    LeftIMGPath.append(row[1].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    RightIMGPath.append(row[2].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




print('Testfile at {}'.format(CenterIMGPath[0]))
image_example = cv2.imread(CenterIMGPath[0],1)
ImgShape =image_example.shape





if (iShowDebugPic==1):



    plt.subplot(231)   
    plt.imshow(LeftImg[30])
    plt.subplot(232)   
    plt.imshow(CenterImg[30])
    plt.subplot(233)   
    plt.imshow(RightImg[30])
    plt.show()
    
   
    
  

#### NN

tf.python.control_flow_ops = tf



while (iShowDebugPic==2):
        #X_batch = []
        #y_batch = []
    batch_size=16
    iIndex = randint(0,len(CenterIMGPath)-batch_size)
        
        #X_batch = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
    X_batch = np.zeros([batch_size,2*64,64,ImgShape[2]])
        #LeftImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        #RightImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
    y_batch = np.zeros(batch_size)

    for i in range(iIndex,iIndex+batch_size):
        iSelect = randint(0,2)
        if (iSelect==0):
            tmpImg = cv2.imread(CenterIMGPath[i],1)
                
                #X_batch[i-iIndex] =cv2.imread(CenterIMGPath[i],1)
                
        elif (iSelect==1):
            tmpImg =cv2.imread(LeftIMGPath[i].strip(),1)
                #y_batch[i-iIndex] = SWA_hist[i]-0.2
        elif (iSelect==2):
            tmpImg =cv2.imread(RightIMGPath[i].strip(),1)
               #y_batch[i-iIndex] = SWA_hist[i]+0.2
            

        if (iShowDebugPic==2):
            plt.subplot(231)   
            plt.imshow(tmpImg)
            #plt.show()
            plt.subplot(232)   
            plt.imshow(cv2.resize(tmpImg,(2*64, 64), interpolation = cv2.INTER_CUBIC))
            #plt.show()
            plt.subplot(233)   
            plt.imshow(cv2.resize(tmpImg,(64, 2*64), interpolation = cv2.INTER_CUBIC))
            plt.show()








number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'

# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

model.add(Lambda(lambda x: x/255.-0.5,input_shape=(64, 2*64, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=(64, 2*64, 3)))
model.add(Activation(activation_relu))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
#model.add(Dropout(0.5))
model.add(Activation(activation_relu))


model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate), loss="mse", )



# try to use other model
#model = models.Sequential()
#model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(64, 2*64, 3), activation='relu'))
#model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
#model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
#model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
#model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
#model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
#model.add(core.Flatten())
#model.add(core.Dense(500, activation='relu'))
#model.add(core.Dropout(.5))
#model.add(core.Dense(100, activation='relu'))
#model.add(core.Dropout(.25))
#model.add(core.Dense(20, activation='relu'))
#model.add(core.Dense(1))
#model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')




model.summary()







t1=time.time()

# fits the model on batches with real-time data augmentation:
model.fit_generator(generate_next_batch(),samples_per_epoch=len(CenterIMGPath), nb_epoch=8)




t2=time.time()
print('Time: {}s'.format(t2-t1))
save_model(model)
