# -*- coding: utf-8 -*-
from glob import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    or not os.path.isfile('test.p'):
            
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
        
    return (train_data,test_data)


def load_scaler(filename):
    with open(filename,'rb') as f:
        scaler = pickle.load(f)
        return scaler


def train(X,y):
    hyperparams = {'C':[0.1,1,10],
                   'kernel':['linear']}
    svc = SVC()
    clf = GridSearchCV(svc,hyperparams,verbose=2)
    clf.fit(X,y)
    return clf


def test(model,X,y):
    return round(model.score(X_test,y_test), 4)


def save_model(model,filename):
    with open(filename,'wb') as f:
        pickle.dump(model, f)
        

def load_model(filename):
    with open(filename,'rb') as f:
        model = pickle.load(f)
        return model
    

def build_window_list(x_range:tuple, y_range:tuple,
                      wndw_sz:tuple, stride:tuple):
    
    wndw_width, wndw_height = wndw_sz[0], wndw_sz[1] 
    
    x_start = x_range[0]
    x_stop  = x_range[1] - (wndw_width-1)
    
    y_start = y_range[0] 
    y_stop  = y_range[1] - (wndw_height-1)
    
    x_stride, y_stride = stride[0], stride[1]
    
    
    # Inclusive range.
    def irange(start,stop,stride):
        return range(start,stop+1,stride)
    
    x_start_pos = irange(x_start,x_stop,x_stride)
    y_start_pos = irange(y_start,y_stop,y_stride)
    
    
    for y_top in y_start_pos:
        y_bottom = y_top + wndw_height -1
        
        for x_left in x_start_pos:
            x_right = x_left + wndw_width - 1
            
            yield [(x_left,y_top),(x_right,y_bottom)]
            
            
def draw_bbox(img,bboxes,color=[0,0,255],thick=5):
    imgcpy = np.copy(img)
    
    for bbox in bboxes:
        cv2.rectangle(imgcpy,bbox[0],bbox[1],color,thick)
    
    return imgcpy


def get_sub_images(img,wndw_sz:tuple,stride:tuple,resize=(64,64)):
    x_range = (0,img.shape[1]-1)
    y_range = (0,img.shape[0]-1)
    
    wndw_list = build_window_list(x_range,y_range,wndw_sz,stride)
    
    for wndw in wndw_list:
        xl,xr = wndw[0][0], wndw[1][0] + 1
        yl,yr = wndw[0][1], wndw[1][1] + 1
        
        sub_image = cv2.resize(img[yl:yr, xl:xr],resize)
        
        yield (sub_image,wndw)
        
        
def window_search(img,wndw_sz:tuple,stride:tuple,model,scaler):
    
    sub_images = get_sub_images(img,wndw_sz,stride)
    
    for sub_image,wndw in sub_images:
        features = get_features(sub_image,colorspace='HSV')
        scaled_features = scaler.transform(features.reshape(1,-1))
        
        is_car = model.predict(scaled_features)
        
        if is_car == 1:
            yield wndw
            
            
def multiscale_window_search(img,wndw_sz_list,strides_list,model,scaler):
       
    for i in range(len(wndw_sz_list)):
        windows = window_search(img,wndw_sz_list[i],strides_list[i],
                                model,scaler)
        for window in windows:
            yield window


def frame_search(img,model,scaler):
    y_start = img.shape[0]//2
    img_roi = img[y_start:,:,:]
    
    wndw_sz_list = [(64,64),(96,72),(128,128),(256,172)]
    strides_list = [(16,16),(32,32),(64,64),(64,32)]
    
    detections = multiscale_window_search(img_roi,
                                          wndw_sz_list,
                                          strides_list,
                                          model,
                                          scaler)
    
    
    for wndw in detections:
        yield[(wndw[0][0],wndw[0][1]+y_start),
              (wndw[1][0],wndw[1][1]+y_start)]
            

if __name__ == '__main__':

    if not os.path.isfile('model.p'):
        train_data,test_data = get_datasets()
        
        X_train, y_train = train_data['data'],train_data['labels']
        X_test,y_test = test_data['data'],test_data['labels']
        
        model = train(X_train, y_train)
        del([train_data])
        
        print('Test Accuracy of SVC = ',test(model,X_test,y_test))
        del([test_data])
    
        save_model(model, 'model.p')
        
    else:    
        model = load_model('model.p')
        
    X_scaler = load_scaler('scaler.p')
        
