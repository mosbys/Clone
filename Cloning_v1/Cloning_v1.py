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

import tensorflow as tf

iShowDebugPic =0

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk
    :param model:
        Keras model to be saved
    :param model_name:
        The name of the model file
    :param weights_name:
        The name of the weight file
    :return:
        None
    """
    #silent_delete(model_name)
    #silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


#sFilePathInput = r'C:\Users\Christoph\Documents\udacity\11_Cloning\simulator-windows-64\driving_log.csv'
sFilePathInput = r'driving_log.csv'
sPathReplace = 'C:\\Users\\Christoph\\Documents\\udacity\\11_Cloning\\simulator-windows-64\\IMG\\'
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
    SWA_hist.append(row[3])
    CenterIMGPath.append(row[0].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    LeftIMGPath.append(row[1].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    RightIMGPath.append(row[2].replace(sPathReplace,str(os.getcwd())+'/IMG/'))
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




print('Testfile at {}'.format(CenterIMGPath[0]))
image_example = cv2.imread(CenterIMGPath[0],1)
ImgShape =image_example.shape

CenterImg = np.zeros([len(CenterIMGPath),ImgShape[0],ImgShape[1],ImgShape[2]])
LeftImg = np.zeros([len(LeftIMGPath),ImgShape[0],ImgShape[1],ImgShape[2]])
RightImg = np.zeros([len(RightIMGPath),ImgShape[0],ImgShape[1],ImgShape[2]])

for i in range(0,len(CenterIMGPath)):
    CenterImg[i] =cv2.imread(CenterIMGPath[i],1)
    LeftImg[i] = cv2.imread(LeftIMGPath[i].strip(),1)
    RightImg[i] = cv2.imread(RightIMGPath[i].strip(),1)




if (iShowDebugPic==1):



    plt.subplot(231)   
    plt.imshow(LeftImg[30])
    plt.subplot(232)   
    plt.imshow(CenterImg[30])
    plt.subplot(233)   
    plt.imshow(RightImg[30])
    plt.show()
    
   
    
  

#### NN
#SWA_hist = SWA_hist[0:100]
CenterIMGPath =CenterIMGPath[0:100]
LeftIMGPath =LeftIMGPath[0:100]
RightIMGPath =  RightIMGPath[0:100]
tf.python.control_flow_ops = tf











number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'

# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

model.add(Lambda(lambda x: x/255.-0.5,input_shape=(160, 320, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=(160, 320, 3)))
model.add(Activation(activation_relu))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Activation(activation_relu))


model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=SGD(learning_rate), loss="mse", )



datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(CenterImg)

# fits the model on batches with real-time data augmentation:
#model.fit_generator(datagen.flow(CenterImg, SWA_hist, batch_size=32),
#                    samples_per_epoch=len(CenterImg), nb_epoch=2)


#for e in range(2):
#    print('Epoch {}'.format(e))
#    batches = 0
#    for X_batch, Y_batch in datagen.flow(CenterImg, SWA_hist, batch_size=32):
#        t1=time.time()
#        loss = model.fit(X_batch, Y_batch)
#        t2=time.time()
#        print('Time: {}s'.format(t2-t1))
#        batches += 1
#        if batches >= len(X_train) / 32:
#            # we need to break the loop by hand because
#            # the generator loops indefinitely
#            break

model.fit(CenterIMG,SWA_hist,32,1)

save_model(model)