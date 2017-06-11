# -*- coding: utf-8 -*-
from glob import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 


def imgread(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def load_image_data(paths):
    data = []
    for path in paths:
        img_data = imgread(path)
        data.append(img_data)
    return np.array(data,dtype=np.uint8)


def extract_spatial_bin_features(img,size=(32,32)):
          
    return cv2.resize(img,size).flatten()


def extract_color_hist_features(img,nbins=32,range_vals=(0,256)):
    chan0_hist = np.histogram(img[:,:,0],bins=nbins,range=range_vals)
    chan1_hist = np.histogram(img[:,:,1],bins=nbins,range=range_vals)
    chan2_hist = np.histogram(img[:,:,2],bins=nbins,range=range_vals)
    
    color_hist_features = np.concatenate((chan0_hist[0],
                                         chan1_hist[0],
                                         chan2_hist[0]))
    return color_hist_features


def extract_hog_features(img_channel, channels=0,nb_orient=9, 
                         nb_pix_per_cell=8,
                         nb_cell_per_block=2, 
                         visualize= False, 
                         ret_vector=True):
    
    if visualize == True:
        features, hog_image = hog(img_channel,orientations=nb_orient,
                                  pixels_per_cell= (nb_pix_per_cell,nb_pix_per_cell),
                                  cells_per_block = (nb_cell_per_block,nb_cell_per_block),
                                  visualise=True,
                                  feature_vector=ret_vector)
        return features, hog_image
    
    else:
        features  = hog(img_channel,orientations=nb_orient,
                                  pixels_per_cell = (nb_pix_per_cell,nb_pix_per_cell),
                                  cells_per_block = (nb_cell_per_block,nb_cell_per_block),
                                  visualise=False,
                                  feature_vector=ret_vector)
        return features
    

def get_features(img,hog_channel='ALL',colorspace='RGB'):
    
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        
        elif colorspace == 'HLS':
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
            
        elif colorspace == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            
        elif colorspace == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
        elif colorspace == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            
        else:
            raise Exception("% colorspace is not a valid colorspace"%(colorspace))
            
    spatial_bin_features = extract_spatial_bin_features(img,
                                                        size=(16,16))
    
    color_hist_features  = extract_color_hist_features(img)
    
    if hog_channel == 'ALL':
        hog_features = [ ]
        
        for channel in range(3):
            hog_features.append(extract_hog_features(img[:,:,channel]))
            
        hog_features = np.ravel(hog_features)
    
    else:
        hog_features = extract_hog_features(img)
    
    return np.concatenate((spatial_bin_features,
                          color_hist_features,
                          hog_features))
    
    
def build_datasets(car_paths,notcar_paths):
    paths = car_paths+notcar_paths
    
    X = []
    for path in tqdm(paths):
        img = imgread(path)
        X.append(get_features(img,colorspace='HSV'))
        
    X = np.reshape(X,[len(paths),-1])
    
    
    y = np.concatenate((np.ones(len(car_paths)),
                       np.zeros(len(notcar_paths))))
    
    
    Scaler_X = StandardScaler().fit(X)    
    
    X_scaled = Scaler_X.transform(X)
    
    X_scaled, y = shuffle(X_scaled,y)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X_scaled,y,train_size=0.7)
        
    del([X_scaled,y])
    
    with open('train.p','wb') as f:
        train_set = {'data':X_train, 'labels':y_train}
        pickle.dump(train_set,f)
        
    with open('test.p','wb') as f:
        test_set = {'data':X_test, 'labels':y_test}
        pickle.dump(test_set,f)
    
    with open('scaler.p','wb') as f:
        pickle.dump(Scaler_X,f)
    
    
    
def get_datasets(force=False):
    
    if (force == True)\
    or not os.path.isfile('train.p')\
    or not os.path.isfile('test.p')\
    or not os.path.isfile('scaler.p'):
            
        # Load all image data.
        vehicle_img_path = []
        vehicle_img_path.extend(glob('vehicles/GTI_Far/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_Left/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_MiddleClose/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_Right/*.png'))
        vehicle_img_path.extend(glob('vehicles/KITTI_extracted/*.png'))
          
        non_vehicle_img_path = []
        non_vehicle_img_path.extend(glob('non-vehicles/Extras/*.png'))
        non_vehicle_img_path.extend(glob('non-vehicles/GTI/*.png'))
        
        build_datasets(vehicle_img_path, non_vehicle_img_path)
        
    with open('train.p','rb') as f:
        train_data = pickle.load(f)
        
    with open('test.p','rb') as f:
        test_data = pickle.load(f)
        
    with open('scaler.p','rb') as f:
        X_scaler = pickle.load(f)
        
    return (train_data,test_data,X_scaler)


def train(X,y):
    svc = LinearSVC()
    svc.fit(X,y)
    return svc


def test(model,X,y):
    return round(model.score(X_test,y_test), 4)


def save_model(model,filename):
    with open(filename,'wb') as f:
        pickle.dump(model, f)
        

def load_model(filename):
    with open(filename,'rb') as f:
        model = pickle.load(f)
        return model



if __name__ == '__main__':

    
    train_data,test_data,X_scaler = get_datasets()
    
    X_train, y_train = train_data['data'],train_data['labels']
    X_test,y_test = test_data['data'],test_data['labels']
    
    model = train(X_train, y_train)
    del([train_data])
    
    print('Test Accuracy of SVC = ',test(model,X_test,y_test))
    del([test_data])
    
    save_model('model.m')
    
    
    
