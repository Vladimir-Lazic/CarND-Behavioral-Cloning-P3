# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Model Visualization"
[image2]: ./examples/training_image.jpg "Training Image"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
For the model I have implemented the NVidia model for End-to-End driving. 

The first layer of the model is a Lamba layer for data normalization. 
The normalization is done in order to help the model generalize the data. For each input image the Lamba layer divides each pixel 
by 255 and subtracts 0.5. With this we have achieved image normalization.

The second layer is a Cropping2D layer. Based on the analysis of the training data, we can see that the top part of the picture is 
just the landscape part of the simulator which has no information value for training the model, so the cropping layer removes it.

![alt text][image2]

Then the network consists of several convolutional layers. The convolutional layers are there to introduce non-linearity to the model. First we
have three Conv2D layers with the kernel width and height of 5x5, then two layers of kernel width and height of 3x3. The depth of each kernels is different, except for the last two which is the same.

Fowlloing the Conv2D layers we have a Flatten layers followed by 3 fully connected layers, with 100, 50 and 10  neuron respectively in each layer. 

The last layer is the output layer with one neuron which represents out steering controls. 

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

In order to reduce te model overfitting and better generalizing the data I tried several things.

The normalization layers in the network architecutre is a help to reducing overfitting as models train better on normalized data.

For the training data I used the provided simulator data. I feed the model the center, left and right image. I also augment the 
center image by flipping it and invertin the measurements. The combination of camera angles and augmentation of center image
is a step in reducing the model overfitting. 

The second thing is the number of epoch used. Since the network used has a powerfull architecture I have trained the network
only for 3 epoch, wich proved to be enough for the model to generalize the data and not too much so it overfitts. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The only parameter tuning was setting 
the number of epochs, which was set to 3.

#### 4. Appropriate training data

For training data I used the provided simulator data, with added augmentations in order to help the model
better generalize the data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model applied to the project is NVidia model for self driving cars. The model proved to be a powerfull architecture 
and was more than capable of generalizing the data. 

While trying to improve the model accuracy in driving, I found that it was not enough to use the camera image from the center 
camera. The model benefited from more training data. So a comibination of center, left and right camera images was applied,
as well as augmentation of center camera image.

For validaation data 20% of the training set was split.

Thre training over 3 epoch resultet in 0.99 validation loss.

