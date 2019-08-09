# digits-recognition-mnist
Project:
Digits handwritten recognition for MNIST based on LeNet model with MATLAB platform.
This project implemented the recognition of handwritten Digits based on LeNet model with MATLAB platform. The structure of the LeNet model was "convolution layer 1 + pooling layer 1 + convolution layer 2+ pooling layer 2 + full connection layer + softmax output layer", the network structure was complete, and the ideal recognition result was achieved.

Requirements：MATLAB

Usage：
Step 1: Run tobmp.m script to generate the appropriate data set for this project;
Step 2: Run train.m to train this model. It is very vital to adjust some parameters, such the inital weights, bias of the convolution kernel;
Step 3: the LeNet_test.m was used for testing.

Results:
The upload file contains a small number of images, and you can get the following recognition results. 
74.44%
If you want to use all of the data from MNIST for experiment, just use the tobmp.m script to get all the data sets, and i’m sure the results will be better.


Notes:
This projects aims to construct a CNN model with MATLAB, the forward and backward algorithms had been archived in this code, and the results showed that it worked, it was for beginers of CNN, if you want to get very good recognition rates of MNIST, please use tensorflow or keras.
