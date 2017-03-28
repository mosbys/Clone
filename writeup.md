#**Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: 00_OrginalImg.jpg "Orignal Image"
[image2]: 01_CropedImg.jpg "Croped image"
[image3]: 02_ReizedWithSWAoffset.jpg "Resized Image with recovery offset"
[image4]: 03_FlippedImg.jpg "Fliped image for variation"
[image5]: 04_ShiftetImg.jpg "Shifted image for more variation and recovery"
[image6]: 05_BrightnessImg.jpg "Brightness changed for training of different variation"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 - > latest version not on github -> too large with 76 MB
* model.json
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the nvidia approach of "End to End Learning for Self-Driving Cars" https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The model consits of a multi layer convolutional network and some fully connected layers.

First input is a 40 x 160 x 3 image to a 5x5 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

Second and third layers are also a   5x5 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

The fourth and fifth layer is a 3x3 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

After this the output is flatten to a fully connected layer (1164), also with a relu activation function. The following layers are reducing to the output size 1 (100, 50, 10,1).

As optimizer is the ADAM optimizer used.



####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 376 / 381 / 386 /395 / ). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ue the nvidia model.

My first step was to use a convolution neural network model similar to the nvidia model. I thought this model might be appropriate because it was proving with real data and a real driving car. Also it was a good motivaiton to understand the technology nivida is using.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there is a high use od dropouts and high variation on input data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve this behavioir I shifted the images and created a virtual steering wheel angle to fake recovery situation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, mostly.

####2. Final Model Architecture

The final model architecture (model.py lines 370 - 418) consisted of a convolution neural network with the following layers and layer sizes:

The model consits of a multi layer convolutional network and some fully connected layers.

First input is a 40 x 160 x 3 image to a 5x5 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

Second and third layers are also a   5x5 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

The fourth and fifth layer is a 3x3 convolulation layer with a relu activation, followed by a Dropout and MaxPooling filter.

After this the output is flatten to a fully connected layer (1164), also with a relu activation function. The following layers are reducing to the output size 1 (100, 50, 10,1).

A visualization can be found at nVidia page (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

For agumention the images and situation:
- I used the orginal image (left / center / right) out of the simlator
![alt text][image1]

- The orginal image was croped to the region of interest, to remove influence of the sky
![alt text][image2]

- Then the images were resized a on left / right images were a virtual steeringwheel angle added respectively substracted
![alt text][image3]

- For a hihger variation all images were randomly flipped and steering angle sign reveresed

![alt text][image4]

- For a better recovery and relation of images on goning straight versus turning, images were shifted randomly.
![alt text][image5]

- To have also a variation on input for different light on the track, the brightness were also changed randomly on some images

![alt text][image6]



I finally randomly shuffled the data set and putted it into batches 


Finally there is a result with a good driving behavoir, but it is a high complex project and there is still a lot of issues to fix. But to get a first touch with behavoir cloning it is a very good project. A lot of input, help and support was used from other repository and slackline. 


