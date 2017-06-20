# -*- coding: utf-8 -*-
from collections import deque
from glob import glob
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
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

GLOBAL_CONFIG = {'SAMPLE_SZ':(64,64) ,
          'COLORSPACE':'HLS',
          'SPATIAL_BIN_SZ':(16,16),
          'COLOR_BINS':32,
          'COLOR_VAL_RANGE':(0,256),
          'HOG_CHANNEL':'ALL',
          'HOG_ORIENTS':9,
          'HOG_PIX_PER_CELL':8,
          'HOG_CELLS_PER_BLOCK':2,
          'CELLS_PER_STEP':2,
          'FRAME_HIST_COUNT':6
        }




def imgread(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def load_image_data(paths):
    data = []
    for path in paths:
        img_data = imgread(path)
        data.append(img_data)
    return np.array(data,dtype=np.uint8)


def extract_spatial_bin_features(img,size):
          
    return cv2.resize(img,size).flatten()


def extract_color_hist_features(img,nbins,range_vals):
    chan0_hist = np.histogram(img[:,:,0],bins=nbins,range=range_vals)
    chan1_hist = np.histogram(img[:,:,1],bins=nbins,range=range_vals)
    chan2_hist = np.histogram(img[:,:,2],bins=nbins,range=range_vals)
    
    color_hist_features = np.concatenate((chan0_hist[0],
                                         chan1_hist[0],
                                         chan2_hist[0]))
    return color_hist_features


def extract_hog_features(img_channel,nb_orient, 
                         nb_pix_per_cell,
                         nb_cell_per_block, 
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
    
    
def cvtColor(img,colorspace:str):
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

    return img
    

def get_features(img,hog_channel,colorspace):
    
    if colorspace != 'RGB':
        img = cvtColor(img,colorspace)
        
            
    spatial_bin_features = \
    extract_spatial_bin_features(img,size =  GLOBAL_CONFIG['SPATIAL_BIN_SZ'])
    
    color_hist_features  = \
    extract_color_hist_features(img,
                                nbins=GLOBAL_CONFIG['COLOR_BINS'],
                                range_vals=GLOBAL_CONFIG['COLOR_VAL_RANGE'])
    
    if hog_channel == 'ALL':
        hog_features = [ ]
        
        for channel in range(3):
            hog_features.append(
                    extract_hog_features(
                            img[:,:,channel],
                            nb_orient=GLOBAL_CONFIG['HOG_ORIENTS'],
                            nb_pix_per_cell=GLOBAL_CONFIG['HOG_PIX_PER_CELL'],
                            nb_cell_per_block = GLOBAL_CONFIG['HOG_CELLS_PER_BLOCK']))
            
        hog_features = np.ravel(hog_features)
    
    else:
        hog_features = extract_hog_features(img[:,:,hog_channel])
    
    return np.concatenate((spatial_bin_features,
                          color_hist_features,
                          hog_features))
    
    
def build_datasets(car_paths,notcar_paths):
    paths = car_paths+notcar_paths
    
    X = []
    for path in tqdm(paths):
        img = imgread(path)
        X.append(get_features(img,
                              hog_channel= GLOBAL_CONFIG['HOG_CHANNEL'],
                              colorspace=GLOBAL_CONFIG['COLORSPACE']))
        
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


def get_sub_images(img,wndw_sz:tuple,stride:tuple,resize=None):
    x_range = (0,img.shape[1]-1)
    y_range = (0,img.shape[0]-1)
    
    wndw_list = build_window_list(x_range,y_range,wndw_sz,stride)
    
    for wndw in wndw_list:
        xl,xr = wndw[0][0], wndw[1][0] + 1
        yl,yr = wndw[0][1], wndw[1][1] + 1
        
        if resize != None:
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


def fast_frame_search(img,y_top,y_bot,scale,model,scaler):
    ''' 
    Perfrom fast search for vehicles across a single video
    frame.
    '''
#    draw_img = np.copy(img)
      
    # Convert colorspace if needed.
    if GLOBAL_CONFIG['COLORSPACE'] != 'RGB':
        img = cvtColor(img,GLOBAL_CONFIG['COLORSPACE'])
    
    # Extract region-of-interest(ROI).    
    img_roi = img[y_top:y_bot,:,:]
    
    # Scale image if needed.
    if scale != 1:
        target_width = np.int(img_roi.shape[1]/scale)
        target_height = np.int(img_roi.shape[0]/scale)
        img_roi = cv2.resize(img_roi,(target_width,target_height))
        
    # Find HOG feature(s) for the entire ROI.
    hog_roi = []
    
    channel_ids = [0,1,2] if GLOBAL_CONFIG['HOG_CHANNEL'] == 'ALL' \
    else [GLOBAL_CONFIG['HOG_CHANNEL']]
    
    for channel_id in channel_ids:
        hog_chann = extract_hog_features(img_roi[:,:,channel_id],
                     nb_orient=GLOBAL_CONFIG['HOG_ORIENTS'],
                     nb_pix_per_cell=GLOBAL_CONFIG['HOG_PIX_PER_CELL'],
                     nb_cell_per_block = GLOBAL_CONFIG['HOG_CELLS_PER_BLOCK'],
                     ret_vector=False)
        hog_roi.append(hog_chann)
        

    
    # Calculate sliding window parameters.
    pix_per_cell = GLOBAL_CONFIG['HOG_PIX_PER_CELL']
    cells_per_block = GLOBAL_CONFIG['HOG_CELLS_PER_BLOCK']
    
    nb_cells_x = (img_roi.shape[1] // pix_per_cell)
    nb_blocks_x = nb_cells_x - (cells_per_block-1)
    
    nb_cells_y = (img_roi.shape[0] // pix_per_cell)
    nb_blocks_y = nb_cells_y - (cells_per_block-1)
    
    wndw_sz = GLOBAL_CONFIG['SAMPLE_SZ'][0]
    blocks_per_window = (wndw_sz // pix_per_cell) - (cells_per_block-1)
    
    cells_per_step = GLOBAL_CONFIG['CELLS_PER_STEP']
    
    nb_steps_x = (nb_blocks_x - blocks_per_window) // cells_per_step
    nb_steps_y = (nb_blocks_y - blocks_per_window) // cells_per_step
    
    for step_y in range(nb_steps_y):
        for step_x in range(nb_steps_x):
            
            cell_x = step_x * cells_per_step
            cell_y = step_y * cells_per_step
            
            hog_features = np.empty([0])
            
            for channel_id in channel_ids:
                hog_chann = hog_roi[channel_id]
                hog_vector = hog_chann[ cell_y:cell_y+blocks_per_window,cell_x:cell_x+blocks_per_window].flatten()
                hog_features = np.hstack((hog_features,hog_vector))
                
                
            xleft_px = cell_x * pix_per_cell
            ytop_px  = cell_y * pix_per_cell

            
            # Extract sub image
            sub_img = img_roi[ytop_px:ytop_px+wndw_sz, xleft_px:xleft_px+wndw_sz]
            
            spatial_features = extract_spatial_bin_features(sub_img,GLOBAL_CONFIG['SPATIAL_BIN_SZ'])
            hist_features    = extract_color_hist_features(sub_img,
                                                           GLOBAL_CONFIG['COLOR_BINS'],
                                                           GLOBAL_CONFIG['COLOR_VAL_RANGE'])
            
            feature_vector = np.hstack((spatial_features,hist_features,hog_features))
            
            scaled_features = scaler.transform(feature_vector.reshape([1,-1]))
            
            prediction = model.predict(scaled_features)
            
            if prediction == 1:
                
                bbox_x_left = np.int(xleft_px*scale)
                bbox_y_top  = np.int(ytop_px*scale)
                
                bbox_sz = np.int(wndw_sz*scale)
                
                yield [(bbox_x_left, bbox_y_top+y_top),(bbox_x_left+bbox_sz, bbox_y_top+y_top+bbox_sz)]
                
#==============================================================================
#                 cv2.rectangle(draw_img,
#                               (bbox_x_left, bbox_y_top+y_top),
#                               (bbox_x_left+bbox_sz, bbox_y_top+y_top+bbox_sz),
#                               (0,0,255),
#                               5)
#==============================================================================
                
#    return draw_img


def add_heat(hmap,bboxes):
    for [(xl,yt),(xr,yb)] in bboxes:
        hmap[yt:yb,xl:xr] += 1
        
        
def remove_heat(hmap,bboxes):
    for [(xl,yt),(xr,yb)] in bboxes:
        hmap[yt:yb,xl:xr] -= 1
    
    
def threshold_heat(hmap,thresh):
    return (hmap >= thresh).astype(np.uint8)


def search_vehicles(img,model,scaler):
    bboxes = []
    
    rois = [(420,550),(420,600),(420,600),(420,620),(420,650),(420,650)]
    scales = [1,1.5,2,2.5,3,3.5]
    
    for i in range(len(scales)):
        bboxes.extend(fast_frame_search(img,rois[i][0],rois[i][1],scales[i],model,scaler))
        
    return bboxes


def labels_to_bboxes(labels):
    nb_labels = labels[1]
    label_map = labels[0]
    
    for label_id in range(1,nb_labels+1):
        nonzero = (label_map == label_id).nonzero()
        
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        xl = np.min(nonzerox)
        xr = np.max(nonzerox)
        
        yt = np.min(nonzeroy)
        yb = np.max(nonzeroy)
        
        yield [(xl,yt),(xr,yb)]
            
            
            

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
    
    
    img_width = 1280
    img_height = 720
    heatmap = np.zeros([img_height,img_width])

    bbox_queue = deque()  
    
    def process_frame(img):
        bboxes_incoming = search_vehicles(img,model,X_scaler)
        
        bboxes_outgoing = []
        
        if len(bbox_queue) > GLOBAL_CONFIG['FRAME_HIST_COUNT']:
            bboxes_outgoing = bbox_queue.popleft()
            
        add_heat(heatmap,bboxes_incoming)
        remove_heat(heatmap,bboxes_outgoing)
        
        high_heat = threshold_heat(heatmap,4)
        
        labels = label(high_heat)
        
        bboxes_to_draw = labels_to_bboxes(labels)
        
        return draw_bbox(img,bboxes_to_draw)
    
    # Process video.
    in_clip = VideoFileClip('test_video.mp4',audio=False)
    out_filename = 'processed-test_video.mp4'
    
    out_clip = in_clip.fl_image(process_frame)
    out_clip.write_videofile(out_filename,audio=False)
