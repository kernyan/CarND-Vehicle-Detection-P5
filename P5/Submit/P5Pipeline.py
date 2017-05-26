# -*- coding: utf-8 -*-

from utils import *   # utility file for this Project - Vehicle Detection
from utilsP4 import * # utility file from Project 4 - Lane Finding
import pickle
from moviepy.editor import VideoFileClip

# Hyperparameters
COLOR_SPACE    = 'YCrCb'
ORIENTS        = 9
PIX_PER_CELL   = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL    = 'ALL'
SPATIAL_SIZE   = (32,32)
HIST_BINS      = 32
SPATIAL_FEAT   = True
HIST_FEAT      = True
HOG_FEAT       = True

HOGPARAMS = {'COLOR_SPACE'   :COLOR_SPACE,
             'ORIENTS'       :ORIENTS,
             'PIX_PER_CELL'  :PIX_PER_CELL,
             'CELL_PER_BLOCK':CELL_PER_BLOCK,
             'HOG_CHANNEL'   :HOG_CHANNEL,
             'SPATIAL_SIZE'  :SPATIAL_SIZE,
             'HIST_BINS'     :HIST_BINS,
             'SPATIAL_FEAT'  :SPATIAL_FEAT,
             'HIST_FEAT'     :HIST_FEAT,
             'HOG_FEAT'      :HOG_FEAT}

RED   = (255,0,0)
GREEN = (0,255,0)
BLUE  = (0,0,255)

YSTART    = 400
YSTOP     = 656
SCALE     = 1
COLOR     = BLUE
THICKNESS = 6

WINPARAMS = {'YSTART'   :YSTART,
             'YSTOP'    :YSTOP,
             'SCALE'    :SCALE,
             'COLOR'    :COLOR,
             'THICKNESS':THICKNESS}

WINPARAMS2 = dict(WINPARAMS)
WINPARAMS2['SCALE'] = 2

Scaler        = pickle.load(open('X_scaler.p', 'rb'))
Clf           = pickle.load(open('svc.p', 'rb'))
Mtx           = pickle.load(open('Mtx.p', 'rb'))
Dist          = pickle.load(open('Dist.p', 'rb'))

FIFOBoxes1      = FIFOBoxes(10)
def Combine_P4P5(img):    
    Labels, Heatmap = P5PipelineRobust(img, WINPARAMS, WINPARAMS2,
                                       HOGPARAMS, Scaler, Clf, FIFOBoxes1)
    ImgDiag4 = DStackSingleChannelImg(Heatmap)
    MaxHeat = np.max(ImgDiag4[...,0])
    if MaxHeat != 0:
        ImgDiag4[...,0] *= int(255/np.max(ImgDiag4[...,0]))
    ImgOutP4, ImgDiag1, ImgDiag2, ImgDiag3 = P4Pipeline(img, Mtx, Dist, MagicCorner_ProjectVideoSDC())
    draw_img = draw_labeled_bboxes(np.copy(ImgOutP4), Labels, WINPARAMS)
    return plot_diagnostics(ImgDiag1, ImgDiag2, ImgDiag3, ImgDiag4, draw_img)    


Output_Path = 'project_video_out_Diag123.mp4'
clip1       = VideoFileClip('project_video.mp4')
VideoOut    = clip1.fl_image(Combine_P4P5)
VideoOut.write_videofile(Output_Path, audio=False)

    
    
    
    
    
    
    
    
    
    
    
    
    