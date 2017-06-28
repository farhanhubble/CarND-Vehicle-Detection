# Udacity Self Driving Car Nanodegree: Computer Vision Project - Vehicle Detection
![Vehicle bounding boxes](output.gif)

This repository contains code for Project 5 from the first term of [Udacity Self-driving Car Nanodegree](https://in.udacity.com/course/self-driving-car-engineer-nanodegree--nd013/). The goal of the project was to detect vehicles in videos, captured by a car camera, using computer vision and machine learning techniques. The code for the entire vehicle detection pipeline is in the file [detect.py](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py). Here's how the pipeline works:

## Generating Training Data from Raw Images

The pipeline uses a SVM classifier to classify vehicles and non-vehicles. The training data comes from Udacity-provided [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. The images are RGB and 64x64px is size. When [detect.py](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py) is run for the first time, the function [`build_datasets()`](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py#L149) is called. This function extracts appropriate features from all the training images, scales the features and creates a feature matrix and a label vector. These are then partitoned into a training set and a test set and saved to **train.p** and **test.p**. The exact features to use are picked to maximize the classifier's accuracy while also keeping the prediction time reasonable, as descibed below.


## Choosing The Feature Space

The images in the dataset were all 64x64 px. As suggested in the lectures, a combination of histogram of oriented gradients (HOG), raw pixel color values (spatial binning) and histogram of pixel values were used as features. 

The classifier was trained using **RGB**, **HSV** and **YCrCb** colorspaces and maximum prediction accuracy was observed for **YCrCb** colorspace, and this colorspace was used in the fina solution. All three channels were used for HOG calculation, spatial binning and for computing histogram of intensities. The parameters were configured via a dictionary **GLOBAL_CONFIG**, defined at the top of the file. 

For HOG, a cell was defined to be 16x16 px. and the number of cells in bock was kept to just 1. Although, 8x8 px. cells and 2x2 blocks performed better, this particular choice was made to cut down the runtime. For spatial binning images were downsampled to 32x32 px. While histogram of intensities was calcuated with 32 bins. Feature extraction is done by [`get_features()`](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py#L111)

## Hyper-parameter selection

A linear SVM classifier was used since non-inear kernel was too slow to cross-validate and because a linear classifier could achieve > 99% accuracy on the test set. The hyper-parameters `C` was found using 3-fold cross validation (CV) using Grid Search. In the code `train()` function performs CV and saves the best model to **model.p**.






