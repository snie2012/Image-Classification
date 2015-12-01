# Image-Classification
Project for CSC522 2015 Fall

Torch:
======================================================= 
Running Instructions:
a) Hardware requirements: High performance GPU and at least 1.5G memory of the GPU.

b) Follow the instructions on http://torch.ch/docs/getting-started.html#_ to install lua and torch. If installed successively, run the command 'th' in the terminal will open the console for torch.

c) Use luarocks(package manager for torch) to install package 'nn' and 'cunn'.

d) Run prepareData.lua, trainModel.lua, testModel.lua in order, then the results will be save in 'results/' folder.

BaseLine:
======================================================= 
The baseline file contains two method in two files:

1. PCA+MLR contains the baseline method of PCA with MLR. In order to run PCA + MLR.py script in Python, the following packages are needed:sklearn, cPickle.

2. HOG+SVM contains PicTransform.m and HOG+SVM.py. PicTransform.m is a matlab script used to transform the mat data to picture data and no addiotianl matlab library is needed. HOG+SVM.py is a python script, in order to run this script, the following python packages are needed, simplecv, opencv, scipy, python imaging library. 
