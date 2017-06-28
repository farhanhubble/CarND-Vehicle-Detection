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

A linear SVM classifier was used since non-inear kernel was too slow to cross-validate and because a linear classifier could achieve > 99% accuracy on the test set. The hyper-parameters `C` was found using 3-fold cross validation (CV) using Grid Search. In the code [`train()`](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py#L225) function performs CV and saves the best model to **model.p**.


## Detecting Vehicles

The core logic for vehicle detection in an image is contianed in the function [`fast_frame_search()`](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py#L336). The function impements a sliding window across an image and uses the SVM model to detect vehicles inside the window only. The region-of-interest (ROI) , that is the part of an image where sliding window search is to perfromed, and the size of the sliding window are passed as parmeters to this function. The search was constrained to the lower half of the image (y=400 ~ y=700) and a window size of 64x64 px was used because the SVM was trained on 64x64 px images.

To perform fast search this function finds the HOG of the entire image at once, then slides a window across the image and calcultes the HOG of the windowed part by sub-sampling the original HOG. To do this the window stride is kept a multiple of the HOG cell size. A stride of 2 cells per step gave good results.

To detect vehicles of different sizes, this function is invoked with different scling factors. To detect vehicles larger thana 64x64 px, the ROI is scaled down by the scaling factor and then a standard 64x64 sliding window search is perfomred. For example if a scaling factor of 2 is passed, the ROI is scaled by a factor of 2 in both directions, so a vehicle that is nearly 128x128 px in the original ROI now resizes to 64x64 px and would be discoverd by a 64x64 sliding window.

This multiscale search is perfomred by [`search_vehicles()`](https://github.com/farhanhubble/CarND-Vehicle-Detection/blob/e6d33a9f870b057c3ab46f6587f3fa4d4422504c/detect.py#L445). The scales were decided considering prediction speed and possible vehicle sizes in the video.






