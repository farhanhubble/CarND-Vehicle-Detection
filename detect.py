# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def imgread(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def load_image_data(paths):
    data = []
    for path in paths:
        img_data = imgread(path)
        data.append(img_data)
    return np.array(data,dtype=np.uint8)


def extract_spatial_bin_features(img,colorspace='RGB',size=(32,32)):
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
            
    return cv2.resize(img,size).flatten()


def extract_color_hist_features(img,nbins=32,range_vals=(0,256)):
    chan0_hist = np.histogram(img[:,:,0],bins=nbins,range=range_vals)
    chan1_hist = np.histogram(img[:,:,1],bins=nbins,range=range_vals)
    chan2_hist = np.histogram(img[:,:,2],bins=nbins,range=range_vals)
    
    color_hist_features = np.concatenate(chan0_hist[0],
                                         chan1_hist[0],
                                         chan2_hist[0])
    return color_hist_features



        
    


def prepare_data():
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
    
    vehicle_imgs = load_image_data(vehicle_img_path)
    
    non_vehicle_imgs = load_image_data(non_vehicle_img_path)
    print(len(vehicle_imgs),len(non_vehicle_imgs))
    
    
        
        
