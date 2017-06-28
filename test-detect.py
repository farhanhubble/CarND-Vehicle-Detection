# -*- coding: utf-8 -*-
from glob import glob
from moviepy.editor import VideoFileClip
from tqdm import tqdm

import cv2
import detect
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_frames(filename,step=1):
    os.makedirs('frames',exist_ok=True)
    clip_in = VideoFileClip(filename,audio=False)
    
    nb_frames = int(clip_in.fps * clip_in.duration)
    
    print('Extrating frames:')
    for i in tqdm(range(0,nb_frames,step)):
        time  = i/clip_in.fps
        clip_in.save_frame('frames/frame{}.png'.format(i),time)
        
        
if __name__ == '__main__':
 #   extract_frames('project_video.mp4',1)
    
    filenames = glob('frames/*.png')
    
    model = detect.load_model('model.p')
    scaler = detect.load_scaler('scaler.p')
    
    img_width = 1280
    img_height = 720
   
    
    os.makedirs('frames/boxes',exist_ok=True)
    os.makedirs('frames/hmaps',exist_ok=True)
    print('Drawing bouding boxes:')
    for filename in tqdm(filenames):
        
        img = detect.imgread(filename)
        bboxes = detect.search_vehicles(img,model,scaler)
        drawn = detect.draw_bbox(img,bboxes)
        path,name = os.path.split(filename)
        cv2.imwrite(path+'/boxes/'+name,cv2.cvtColor(drawn,cv2.COLOR_RGB2BGR))
        
#        heatmap = np.zeros([img_height,img_width])
#        detect.add_heat(heatmap,bboxes)
#        plt.imshow(heatmap,cmap='hot')
#        plt.savefig(path+'/hmaps/'+name)
        
        
    


