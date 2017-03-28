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
import random
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling


from random import randint
import tensorflow as tf
import json

iShowDebugPic =7
newShape=np.array(3)

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to half size
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = image[top:-bottom, :]
    newShape = image.shape
    image= cv2.resize(image,(int(newShape[1]/2), int(newShape[0]/2)), interpolation = cv2.INTER_CUBIC)
    return image

def trans_image(image,steer,trans_range):
    # Translation of image - move to right and left
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1],image.shape[0]))

    return image_tr,steer_ang,tr_x


def generate_next_batch(batch_size=256):
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
        #iIndex = randint(0,len(CenterIMGPath)-batch_size)
        
        #X_batch = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        X_batch = np.zeros([batch_size,newShape[0],newShape[1],ImgShape[2]])
        #LeftImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        #RightImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        y_batch = np.zeros(batch_size)

        #for i in range(iIndex,iIndex+batch_size):
        i=0
        while (i<batch_size):
            iSelect = randint(0,2)
            iIndex = randint(0,len(CenterIMGPath)-batch_size)
            if (iSelect==0):
                tmpImg = cv2.imread(CenterIMGPath[iIndex],1)
                y_batch[i] = SWA_hist[iIndex]
                #X_batch[i-iIndex] =cv2.imread(CenterIMGPath[i],1)
                
            elif (iSelect==1):
                tmpImg =cv2.imread(LeftIMGPath[i].strip(),1)
                y_batch[i] = SWA_hist[iIndex]+0.25
            elif (iSelect==2):
                tmpImg =cv2.imread(RightIMGPath[i].strip(),1)
                y_batch[i] = SWA_hist[iIndex]-0.25
            

            if (iShowDebugPic==2):
                plt.subplot(321)   
                plt.imshow(tmpImg)
                #plt.show()
                plt.subplot(322)   
                plt.imshow(cv2.resize(tmpImg,(2*64, 64), interpolation = cv2.INTER_CUBIC))
                #plt.show()
                plt.subplot(323)   
                plt.imshow(preprocess(tmpImg))
                
                plt.subplot(324)   
                plt.imshow(cv2.resize(preprocess(tmpImg),(2*64, 64), interpolation = cv2.INTER_CUBIC))

                plt.subplot(325)   
                tmpImg2 = tmpImg[ :, ::-1, :]
                plt.imshow(tmpImg2)
                plt.subplot(326)   
                plt.imshow(cv2.resize(preprocess(tmpImg2),(2*64, 64), interpolation = cv2.INTER_CUBIC))
                
                plt.show()
               
            iFlipImg = randint(0,4)
            if (iFlipImg>1):
                tmpImg = tmpImg[ :, ::-1, :]
                y_batch[i] = -y_batch[i]
            tmpImg = cv2.cvtColor(tmpImg,cv2.COLOR_BGR2RGB)
            tmpImg=preprocess(tmpImg)
            tmpImg = augment_brightness_camera_images(tmpImg)
            if ((iFlipImg==0) or (iFlipImg==4)):

                if (iSelect==10):  
                    test3=(tmpImg)
                if (iSelect==11):  
                    tmpImg,y_batch[i],tr_x  =  trans_image(tmpImg,y_batch[i],50)
                if (iSelect==12):  
                    tmpImg,y_batch[i],tr_x  =  trans_image(tmpImg,y_batch[i],-50)
            tmpImg,y_batch[i],tr_x  =  trans_image(tmpImg,y_batch[i],100)    
            X_batch[i] = tmpImg
            #newShape = X_batch[i].shape
            
            #X_batch[i] = cv2.resize(tmpImg,(newShape[1], newShape[0]), interpolation = cv2.INTER_CUBIC)
            if ((abs(y_batch[i])<0.001) & (iFlipImg==1)):
                i=i
            else:
                i=i+1
                
            #y_batch[i-iIndex] = SWA_hist[i]

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
Speed_hist=[]
CenterIMGPath = []
LeftIMGPath = []
RightIMGPath = []
File = open(sFilePathInput,'r')
ImgShape =[]
csvInput = csv.reader(File, delimiter=',')
for row in csvInput:
    #print(row)
    #image = cv2.imread(row[0],0)
    if (float(row[6])>20):
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
print('Testfile shape = {} x {} x {}'.format(ImgShape[0],ImgShape[1],ImgShape[2]))
image_example = (preprocess(image_example))
newShape = image_example.shape

print('Testfile will be resized to shape = {} x {} x {}'.format(newShape[0],newShape[1],newShape[2]))




if (iShowDebugPic==1):
    plt.subplot(231)   
    plt.imshow(LeftImg[30])
    plt.subplot(232)   
    plt.imshow(CenterImg[30])
    plt.subplot(233)   
    plt.imshow(RightImg[30])
    plt.show()

if (iShowDebugPic==3):
    plt.hist(SWA_hist)  
    plt.savefig("histSWA.png")
   
    
  

#### NN

tf.python.control_flow_ops = tf



while (iShowDebugPic>6):
        #X_batch = []
        #y_batch = []
    dumpFileName=''
    batch_size=16
    iIndex = randint(0,len(CenterIMGPath)-batch_size)
        
        #X_batch = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
    X_batch = np.zeros([batch_size,2*64,64,ImgShape[2]])
        #LeftImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
        #RightImg = np.zeros([batch_size,ImgShape[0],ImgShape[1],ImgShape[2]])
    y_batch = np.zeros(batch_size)

    for i in range(iIndex,iIndex+batch_size):
        iSelect = randint(0,2)
        iIndex = randint(0,len(CenterIMGPath)-batch_size)
        if (iSelect==0):
            tmpImg = cv2.imread(CenterIMGPath[i],1)
            dumpFileName= CenterIMGPath[i]  
                #X_batch[i-iIndex] =cv2.imread(CenterIMGPath[i],1)
                
        elif (iSelect==1):
            tmpImg =cv2.imread(LeftIMGPath[i].strip(),1)
            dumpFileName=LeftIMGPath[i]
                #y_batch[i-iIndex] = SWA_hist[i]-0.2
        elif (iSelect==2):
            tmpImg =cv2.imread(RightIMGPath[i].strip(),1)
            dumpFileName=RightIMGPath[i]
               #y_batch[i-iIndex] = SWA_hist[i]+0.2
        tmpSWA = SWA_hist[i]    
        tmpImg = cv2.cvtColor(tmpImg,cv2.COLOR_BGR2RGB)
        if (iShowDebugPic==6):
                sp=plt.subplot(421)   
                plt.imshow(tmpImg)
                sp.set_title('Axis [1,1]')
                #plt.show()
                plt.subplot(422)   
                plt.imshow(cv2.resize(tmpImg,(2*64, 64), interpolation = cv2.INTER_CUBIC))
                #plt.show()
                plt.subplot(423)   
                plt.imshow(preprocess(tmpImg))
                
                plt.subplot(424)   
                plt.imshow(cv2.resize(preprocess(tmpImg),(2*64, 64), interpolation = cv2.INTER_CUBIC))

                plt.subplot(425)   
                tmpImg2 = tmpImg[ :, ::-1, :]
                plt.imshow(tmpImg2)
                plt.subplot(426)   
                plt.imshow(cv2.resize(preprocess(tmpImg2),(2*64, 64), interpolation = cv2.INTER_CUBIC))
                v_delta = .05
                tmpImg3 = preprocess(
                    tmpImg,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                plt.subplot(427)
                plt.imshow(preprocess(tmpImg3))
                
                plt.subplot(428)   
                plt.imshow(cv2.resize(preprocess(tmpImg3),(2*64, 64), interpolation = cv2.INTER_CUBIC))



                plt.show()
                tmpImg4 = trans_image(tmpImg,tmpSWA,80)
                plt.subplot(311)   
                plt.imshow(tmpImg)
                #plt.show()
                plt.subplot(312)   
                plt.imshow(tmpImg4[0][0:160][:][:])
                #plt.show()
                plt.subplot(313)   
                plt.imshow(preprocess(tmpImg4[0][0:160][:][:]))
                plt.show()

        if (iShowDebugPic==7):
                sPathReplace = 'C:\\Users\\Christoph\\Documents\\udacity\\11_Cloning\\simulator-windows-64\\IMG\\'
                sp1=plt.subplot(321)
                #plt.title('Orginal Img'+dumpFileName)   
                plt.imshow(tmpImg)
                plt.imsave('00_OrginalImg.jpg',tmpImg)
                sp1.set_title('Orginal Img'+ dumpFileName.replace(sPathReplace,''))
                #plt.show()
                sp2=plt.subplot(322)   
                plt.imshow(preprocess(tmpImg))
                plt.imsave('01_CropedImg.jpg',preprocess(tmpImg))
                sp2.set_title('Croped Img - SWA orginal = '+str(tmpSWA))
                #plt.show()
                sp3=plt.subplot(323)   
                if (iSelect==1):
                    tmpSWA = tmpSWA +0.2
                elif (iSelect==2):
                    tmpSWA = tmpSWA -0.2
                test = (preprocess(tmpImg))
                newShape = test.shape
                #plt.imshow(cv2.resize(preprocess(tmpImg),(newShape[1], newShape[0]), interpolation = cv2.INTER_CUBIC))
                plt.imshow(test)
                plt.imsave('02_ReizedWithSWAoffset.jpg',test)
                sp3.set_title('Resized 40 x 160 - SWA with offset = '+str(tmpSWA) )
                sp4=plt.subplot(324)
                test = test[ :, ::-1, :]   
                #plt.imshow(cv2.resize(preprocess(tmpImg2),(newShape[1], newShape[0]), interpolation = cv2.INTER_CUBIC))
                plt.imsave('03_FlippedImg.jpg',test)
                plt.imshow(test)
                sp4.set_title('Flipped - SWA with offset and flipped= '+str(-tmpSWA) )

                sp5=plt.subplot(325) 
                if (iSelect==0):  
                    test3,y_steer,tr_x  =  trans_image(preprocess(tmpImg),tmpSWA,1)
                if (iSelect==1):  
                    test3,y_steer,tr_x  =  trans_image(preprocess(tmpImg),tmpSWA,50)
                if (iSelect==2):  
                    test3,y_steer,tr_x  =  trans_image(preprocess(tmpImg),tmpSWA,-50)
                plt.imshow(test3)
                sp5.set_title('Shiftet  Img - SWA transformed = '+str(y_steer))
                plt.imsave('04_ShiftetImg.jpg',test3)
                sp6=plt.subplot(326)   
                test4 = augment_brightness_camera_images(preprocess(tmpImg))
                plt.imshow(test4)
                plt.imsave('05_BrightnessImg.jpg',test4)
                sp6.set_title('Changed brightness - final img')

                plt.show()






number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'


# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

#model.add(Lambda(lambda x: x/255.-0.5,input_shape=(newShape[0], newShape[1], 3)))

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(newShape[0], newShape[1], 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=(newShape[0], newShape[1], 3)))
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








model.summary()








t1=time.time()

# fits the model on batches with real-time data augmentation:
SWA_overall = np.zeros(len(CenterIMGPath)*5)
model.fit_generator(generate_next_batch(),samples_per_epoch=len(CenterIMGPath)*10, nb_epoch=5)




t2=time.time()
print('Time: {}s'.format(t2-t1))
save_model(model)
