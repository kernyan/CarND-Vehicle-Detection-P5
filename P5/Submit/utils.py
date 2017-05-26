# -*- coding: utf-8 -*-

import os
import glob
from warnings import warn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from queue import deque



def GetDirectoryFiles(DirName):    
    List = []
    for each in os.listdir(DirName):
        if (each.endswith('jpg') or
            each.endswith('jpeg') or
            each.endswith('png')):
            List.append(DirName+each)
        else:
            # is directory
            List.extend(glob.glob(DirName+each+'/*'))
    return sorted(List)


def WriteListToFile(List, FileName):
    with open(FileName, 'w') as f:
        for each in List:
            f.write(each+'\n')
        

def ReadImage(Path):
    '''
    Returns uint8 image. If is png than scale float to uint8
    NOTE: image is in RGB order
    '''
    FileType = Path.split('.')[-1] 
    if FileType == 'jpg' or FileType == 'jpeg':
        return mpimg.imread(Path)
    elif FileType == 'png':
        return cv2.convertScaleAbs(mpimg.imread(Path), alpha = 255.0)
    else:
        raise NameError('Unknown image extension being read')


def CompareTwoImages(Img1, Img2, 
                     IsImg1Gray=False, IsImg2Gray=False, 
                     Img1Title='Img1', Img2Title='Img2',
                     SaveFileName = None):
    '''
    Displays two images side by side
    '''
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    if IsImg1Gray:
        ax1.imshow(Img1, cmap='gray')
    else:
        ax1.imshow(Img1)
    ax1.set_title(Img1Title, fontsize=30)
            
    if IsImg2Gray:
        ax2.imshow(Img2, cmap='gray')
    else:
        ax2.imshow(Img2)
    ax2.set_title(Img2Title, fontsize=30)
    
    if SaveFileName is not None:
        plt.savefig(SaveFileName)  




def plot3d(pixels, colors_rgb, axis_labels=list("RGB"),
           axis_limits=[(0, 255), (0, 255), (0, 255)]):
    '''
    Plot pixels in 3D
    
    Reference Udacity's SDC Vehicle Detection Section
    '''
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(pixels[:, :, 0].ravel(),
               pixels[:, :, 1].ravel(),               
               pixels[:, :, 2].ravel(),
               c=colors_rgb.reshape((-1, 3)), edgecolors='none')


def Plot3DColorSpace(img, TargetColorSpace, SaveFileName=None):
    '''
    Plots img in 3D for visualization (used to aid Color Space selection)
    
    Reference: Udacity's SDC Vehicle Detection section
    '''
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img,
                           (np.int(img.shape[1] / scale), 
                            np.int(img.shape[0] / scale)),
                            interpolation=cv2.INTER_NEAREST)

    if TargetColorSpace is None:
        img_small_Target = img_small
    else:
        img_small_Target = cv2.cvtColor(img_small, TargetColorSpace)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    plot3d(img_small_Target, img_small_rgb)
    
    plt.title(SaveFileName.split('/')[-1][:-4]) # get file name, exclude file extension
    if SaveFileName is not None:
        plt.savefig(SaveFileName)
    
    
def get_hog_features(img, params, vis = False, feature_vec=True):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    if vis == True:
        features, hog_image = hog(img,
                                  params['ORIENTS'],
                                  (params['PIX_PER_CELL'], params['PIX_PER_CELL']),
                                  (params['CELL_PER_BLOCK'], params['CELL_PER_BLOCK']),
                                  transform_sqrt=False,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img,
                       params['ORIENTS'],
                       (params['PIX_PER_CELL'], params['PIX_PER_CELL']),
                       (params['CELL_PER_BLOCK'], params['CELL_PER_BLOCK']),
                       transform_sqrt=False,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features
    

def bin_spatial(img, size=(32,32)):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    color1 = cv2.resize(img[...,0], size).ravel()
    color2 = cv2.resize(img[...,1], size).ravel()
    color3 = cv2.resize(img[...,2], size).ravel()
    return np.concatenate((color1, color2, color3))
    

def color_hist(img, nbins=32):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    channel1_hist = np.histogram(img[...,0], bins=nbins)
    channel2_hist = np.histogram(img[...,1], bins=nbins)
    channel3_hist = np.histogram(img[...,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    return hist_features
    

def extract_features(Files, params):
    '''
    Extracts features for all files
    '''
    Features = []
    NumberOfFeaturesPerImage = 0
    for k, File in enumerate(Files):
        img = ReadImage(File)
        FeaturesOneFile = extract_single_features(img, params)
        if k == 0: # only need to calculate this for one file
            NumberOfFeaturesPerImage = len(FeaturesOneFile)
        Features.append(FeaturesOneFile)
    
    return Features, NumberOfFeaturesPerImage
     

def ConvertRGBTo(img, color_space):
    '''
    Returns image of target color space
    '''
    if color_space == 'RGB':
        feature_img = np.copy(img)
    elif color_space == 'HSV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)            
    elif color_space == 'LUV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'gray':
        warn('Returned image has different channel count than input image')
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise NameError('Unexpected color channel encountered')
    return feature_img


def extract_single_features(img, params):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''    
    Features = []
    feature_img = ConvertRGBTo(img, params['COLOR_SPACE'])    

    # bin spatial
    if params['SPATIAL_FEAT'] == True:
        feature_spatial = bin_spatial(feature_img, params['SPATIAL_SIZE'])        
        Features.append(feature_spatial)
    
    # color histogram
    if params['HIST_FEAT'] == True:
        feature_hist = color_hist(feature_img, params['HIST_BINS'])
        Features.append(feature_hist)

    # hog feature            
    if params['HOG_FEAT'] == True:
        if params['HOG_CHANNEL'] == 'ALL':
            feature_hogs = []
            for Channel in range(feature_img.shape[2]):
                feature_hogs.append(
                        get_hog_features(feature_img[...,Channel],
                                         params,
                                         vis=False,
                                         feature_vec=True))
            feature_hogs = np.ravel(feature_hogs)
        else:            
            feature_hogs = get_hog_features(feature_img[...,params['HOG_CHANNEL']],
                                            params,
                                            vis=False,
                                            feature_vec=True)        
        Features.append(feature_hogs)
        
    Features = np.concatenate(Features)
    
    return Features



def PrepareTrainTestData(PositiveList, NegativeList, hogparams, TrainTestSplit=0.1):
    '''
    Return training and testing data from list of positive and negative image files
    '''
    t= time.time()
    Positives, NumberOfFeatures = extract_features(PositiveList, hogparams)
    Negatives, _                = extract_features(NegativeList, hogparams)
    print('{0:7} Features took {1:6.2f} to extract'.format(NumberOfFeatures, time.time()-t))

    X        = np.vstack((Positives, Negatives)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(Positives)), np.zeros(len(Negatives))))
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=TrainTestSplit,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def TrainSVC(X_train, X_test, y_train, y_test, Clf):
    '''
    Train SVC classifier
    '''    
    t   = time.time()
    Clf.fit(X_train, y_train)
    print('{0:6.2f} seconds to fit classifier'.format(time.time()-t))
    print('{0:7.4f}% Test accuracyC'.format(Clf.score(X_test, y_test)*100))


def ExtractAndTrainSVC(PositiveImgs, NegativeImgs, hogparams):
    '''
    Prints number of image features, extraction time, classifier fitting time,
    and testing accuracy based on passed in parameter list
    
    hogparams is a paramter dictionary. Declared as HOGPARAMS in P5Pipeline.py
    '''
    X_train, X_test, y_train, y_test = PrepareTrainTestData(PositiveImgs,
                                                            NegativeImgs,
                                                            hogparams,
                                                            TrainTestSplit=0.1)

    svc = LinearSVC()
    TrainSVC(X_train, X_test, y_train, y_test, svc)
    return svc


def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64,64),
                 xy_overlap=(0.5,0.5)):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # search span
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # pixel per step
    nx_pix_per_step = np.int(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1-xy_overlap[1]))
    
    # number of windows
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx   = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy   = starty + xy_window[1]
            
            window_list.append(((startx, starty),(endx, endy)))
            
    return window_list


def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def GetSubRegion(img, window, ImgShape=(64,64)):
    '''
    Returns subregion of img according to window
    option to resize
    '''
    imgView = img[window[0][1]:window[1][1],
                  window[0][0]:window[1][0]]
    
    Width  = window[0][0] - window[1][0]
    Height = window[0][1] - window[1][1]
    
    if  Width == ImgShape[1] and Height == ImgShape[0]:
        return imgView
    else:
        return cv2.resize(imgView, ImgShape)


def search_windows(img, windows, clf, scaler, hogparams):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''
    on_windows = []
    for window in windows:
        test_img = GetSubRegion(img, window, ImgShape=(64,64))
        features = extract_single_features(test_img, hogparams)
        # scale features
        test_features = scaler.transform(
                np.array(features).reshape(1,-1))
        
        # perform prediction
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows
        

def visualize(fig, rows, cols, imgs, titles, SaveFileName=None):
    '''
    Reference: Udacity's SDC Vehicle Detection section
    '''    
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
        else:
            plt.imshow(img)
        plt.title(titles[i])
    if SaveFileName is not None:
        plt.savefig(SaveFileName)  
    

def MarkHeatmap(x, y, WindowSize, Heatmap):
    '''
    Marks detected image on heatmap
    Assumes rectangle bounding boxes
    '''
    Heatmap[y:y+WindowSize, x:x+WindowSize] += 1
    return Heatmap


def MarkHeatMapbyBoxes(Heatmap, Boxes):
    if Boxes is not None:
        for Box in Boxes:
            Heatmap[Box[0][1]:Box[1][1], Box[0][0]:Box[1][0]] += 1
    return Heatmap


def SearchImageHogSubsample(File, winparams, hogparams, Scaler, Clf):
    '''
    Search input image using sliding window, extract features according to passed in parameter list,
    then performs prediction using passed in classifier.

    Output: image with bounding boxes, heatmap, and detection boxes     
    
    Reference Udacity's SDC Vehicle Detection section
    '''
    if type(File) == 'str':
        img = ReadImage(File)
    else:
        img = File    
    boxes     = []    
    count     = 0
    draw_img  = np.copy(img)
    
    # blank heat map
    heatmap       = np.zeros_like(img[...,0])
    ImgCropped    = img[winparams['YSTART']:winparams['YSTOP'],...]
    ImgCvtColor   = ConvertRGBTo(ImgCropped, hogparams['COLOR_SPACE'])
    
    if winparams['SCALE'] != 1:
        imshape     = ImgCvtColor.shape
        ImgCvtColor = cv2.resize(ImgCvtColor,
                                 (int(imshape[1]/winparams['SCALE']),
                                  int(imshape[0]/winparams['SCALE'])))
        
    ch1 = ImgCvtColor[...,0]
    ch2 = ImgCvtColor[...,1]
    ch3 = ImgCvtColor[...,2]
    
    # possible block positions for HOG in x and y direction    
    assert(hogparams['CELL_PER_BLOCK'] == 2)
    kBlocksByImgWidth  = (ch1.shape[1] // hogparams['PIX_PER_CELL']) - 1 # assumes that 2 cells per block
    kBlocksByImgHeight = (ch1.shape[0] // hogparams['PIX_PER_CELL']) - 1

    ClfWindow        = 64 # classifier window size (64, 64)
    kBlocksPerClfWin = (ClfWindow//hogparams['PIX_PER_CELL']) - 1 # also assumes that 2 cells per block
    CellStep         = 2 # move two cell length for ClfWindow over entire image (i.e., 8 * 2 pixels)
    
    kStepsHorizontal = (kBlocksByImgWidth  - kBlocksPerClfWin) // CellStep # number of ClfWindows possible over image width
    kStepsVertical   = (kBlocksByImgHeight - kBlocksPerClfWin) // CellStep
    
    hog1 = get_hog_features(ch1, hogparams, feature_vec=False) # returns feature vec by (block pos, block pos, cells, cells, orientations)
    hog2 = get_hog_features(ch2, hogparams, feature_vec=False)
    hog3 = get_hog_features(ch3, hogparams, feature_vec=False)

    for xStep in range(kStepsHorizontal):
        for yStep in range(kStepsVertical):
            count += 1
            yCellPos = yStep*CellStep # cell position within scaled searchimage
            xCellPos = xStep*CellStep
            
            hog_f1 = hog1[yCellPos:yCellPos+kBlocksPerClfWin, xCellPos:xCellPos+kBlocksPerClfWin].ravel() # picking from [block, block, ...] of feature vector
            hog_f2 = hog2[yCellPos:yCellPos+kBlocksPerClfWin, xCellPos:xCellPos+kBlocksPerClfWin].ravel()
            hog_f3 = hog3[yCellPos:yCellPos+kBlocksPerClfWin, xCellPos:xCellPos+kBlocksPerClfWin].ravel()
            hog_features = np.hstack((hog_f1, hog_f2, hog_f3))
            
            xPixelPos = xCellPos*hogparams['PIX_PER_CELL'] # pixel position within scaled searchimage
            yPixelPos = yCellPos*hogparams['PIX_PER_CELL']
            
            ImgClf = ImgCvtColor[yPixelPos:yPixelPos+ClfWindow, xPixelPos:xPixelPos+ClfWindow]
            assert(ImgClf.shape == (64,64,3)) # image size used to train classifier, must be adhered
            spatial_features = bin_spatial(ImgClf, hogparams['SPATIAL_SIZE'])
            hist_features    = color_hist(ImgClf, hogparams['HIST_BINS'])
            StackedFeatures  = np.hstack((spatial_features,
                                          hist_features,
                                          hog_features)).reshape(1,-1)
            test_features    = Scaler.transform(StackedFeatures)
            test_prediction  = Clf.predict(test_features)
            
            if test_prediction == 1:
                xTruePixelPos = np.int(xPixelPos*winparams['SCALE']) # pixel position on unscaled image
                yTruePixelPos = np.int(yPixelPos*winparams['SCALE'])
                TrueWinSize   = np.int(ClfWindow*winparams['SCALE'])
                RectanglePts  = ((xTruePixelPos, yTruePixelPos+winparams['YSTART']),
                                 (xTruePixelPos+TrueWinSize, yTruePixelPos+TrueWinSize+winparams['YSTART']))
                cv2.rectangle(draw_img, *RectanglePts, winparams['COLOR'], winparams['THICKNESS'])
                boxes.append(RectanglePts)
                
                # mark detected image on heatmap
                MarkHeatmap(xTruePixelPos,
                            yTruePixelPos+winparams['YSTART'],
                            TrueWinSize,
                            heatmap)                
    
    return draw_img, File[-9:], heatmap, boxes
    

def SingleThreshold(Heatmap, thres):
    HeatmapCopy = np.copy(Heatmap)
    HeatmapCopy[HeatmapCopy <= thres] = 0
    return HeatmapCopy


def MultipleThreshold(Heatmaps, thres):
    HeatmapsCopy = np.copy(Heatmaps)
    for Heatmap in HeatmapsCopy:
        Heatmap[Heatmap <= thres] = 0
    return HeatmapsCopy

    
def draw_labeled_bboxes(img, labels, winparams):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], winparams['COLOR'], winparams['THICKNESS'])
    # Return the image
    return img


class FIFOBoxes():
    def __init__(self, n=10):
        self.n = n
        self.queued_boxes  = deque([], maxlen=n)        
        self.allboxes      = []
        
    def Add(self, boxes):
        self.queued_boxes.appendleft(boxes)
    
    def AllBoxes(self):
        allboxes = []
        for boxes in self.queued_boxes:
            allboxes += boxes
        if len(allboxes)==0:
            self.allboxes = None
        else:
            self.allboxes = allboxes
            
    def Update(self, boxes):        
        self.Add(boxes)
        self.AllBoxes()


def P5PipelineSimple(img, winparams, hogparams, Scaler, Clf):
    # hog subregion search
    _, _, Heatmap, Boxes   = SearchImageHogSubsample(img, winparams,  hogparams, Scaler, Clf)
        
    labels   = label(Heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels, winparams)            
    
    return draw_img, Heatmap


def P5PipelineRobust(img, winparams, winparams2, hogparams, Scaler, Clf, FIFO):
     
    # hog subregion search
    _, _, _, Boxes  = SearchImageHogSubsample(img, winparams,  hogparams, Scaler, Clf)
    _, _, _, Boxes2 = SearchImageHogSubsample(img, winparams2, hogparams, Scaler, Clf)    
        
    Boxes.extend(Boxes2)
    FIFO.Update(Boxes)
    
    Heatmap  = np.zeros_like(img[...,0]).astype(np.uint8)
    Heatmap  = MarkHeatMapbyBoxes(Heatmap, FIFO.allboxes)
    Heatmap  = SingleThreshold(Heatmap, 20)
    labels   = label(Heatmap)    
    
    return labels, Heatmap


def plot_diagnostics(image1, image2, image3, image4, image_final):
    '''Plot a number of images created by the pipeline into a single image
    
    Reference: GeoffBreemer
    '''
    diagScreen = np.zeros((1080, 1280, 3), dtype=np.uint8)

    # Main screen
    diagScreen[0:720, 0:1280] = image_final

    # Four screens along the bottom
    diagScreen[720:1080, 0:320]    = cv2.resize(image1, (320,360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 320:640]  = cv2.resize(image2, (320,360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 640:960]  = cv2.resize(image3, (320,360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 960:1280] = cv2.resize(image4, (320,360), interpolation=cv2.INTER_AREA)    

    return diagScreen


def ExtractVideoFrames(clip, Folder, FrameStart, FrameEnd):
    assert(0<FrameStart<1000) # filename convention from 000 to 999 to allow sorting
    assert(0<FrameEnd<1000)
    assert(FrameStart<=FrameEnd)
    countF = 0
    for frame in clip.iter_frames():
        countF += 1
        if countF == 1000:
            break
        if FrameStart <= countF < FrameEnd:
            if countF >= 100:
                str1 = str(countF)
                FileNameSv = '.{}{}{}{}.png'.format(Folder, str1[0], str1[1], str1[2])
            elif countF >= 10:
                str1 = str(countF)
                FileNameSv = '.{}0{}{}.png'.format(Folder, str1[0], str1[1])                
            else:
                str1 = str(countF)
                FileNameSv = '.{}00{}.png'.format(Folder, str1)
            plt.imsave(FileNameSv, frame)        


def VisualizeImg(Files, Count=100, SaveFileName=None):
    ImgList = []
    Title = []
    HeatmapsOut = []
    t = time.time()
    for k, each in enumerate(Files):
        if k == Count:
            break
        Img = ReadImage(each)
        ImgOut, HeatOut = process_imageP5(Img)
        ImgList.append(ImgOut)
        Title.append(each)
        HeatmapsOut.append(HeatOut)
    print('Search took',time.time()-t,'s')    
    fig = plt.figure(figsize=(12,Count*2))
    visualize(fig, int(Count/2)+1,2,ImgList,Title,SaveFileName)
    return ImgList, Title, HeatmapsOut

































































