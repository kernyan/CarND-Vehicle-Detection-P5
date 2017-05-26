# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def GlobDirectory(RegexName):    
    return sorted(glob.glob(RegexName))


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


def FindChessBoardCorners(BoardSize, calibration_images):
    '''
    Returns chess board x,y,z corners
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
    
    x, y = BoardSize
    ObjPoints = [] # xyz where z is zero, undistorted grid locations
    ImgPoints = [] # xyz on distorted chessboards
    
    Objp = np.zeros((x*y, 3), dtype=np.float32)
    Objp[:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1,2)
    
    for Path in calibration_images:
        Img = ReadImage(Path)
        Gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
        Ret, Corners = cv2.findChessboardCorners(Gray, BoardSize, None)
        
        if Ret is True:
            ImgPoints.append(Corners)
            ObjPoints.append(Objp)
            
    return ObjPoints, ImgPoints


def CalibrateCamera(ObjPoints, ImgPoints, ImgShape):
    '''
    Returns camera matrix, and distortion coefficients
    '''    
    
    Ret, Mtx, Dist, Rvecs, Tvecs = cv2.calibrateCamera(ObjPoints, ImgPoints, ImgShape, None, None)
    if Ret:
        return Mtx, Dist
    else:        
        raise ValueError('Camera not calibrated. Review CalibrateCamera()')
    return None


def UndistortImg(Img, Mtx, Dist):
    return cv2.undistort(Img, Mtx, Dist, None, Mtx)  

      
def CompareTwoImages(Img1, Img2, 
                     IsImg1Gray=False, IsImg2Gray=False, 
                     Img1Title='Img1', Img2Title='Img2',
                     SaveFileName = None):
    '''
    Displays two images side by side
    '''
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
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


def Sobel_Binary(Img, Orient='x', Sobel_Kernel=3, Thresh=(0,255)):
    '''
    Returns binary sobel-thresholded image
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
        
    gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)    
    if Orient == 'x':
        SobelImg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=Sobel_Kernel)
    elif Orient == 'y':
        SobelImg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=Sobel_Kernel)
    else:
        raise NameError('Invalid orient case')
            
    Abs_Sobel    = np.absolute(SobelImg)
    Scale_Factor = np.max(Abs_Sobel)/255.0
    Scaled_Sobel = (Abs_Sobel/Scale_Factor).astype(np.uint8)
    Binary       = np.zeros_like(Scaled_Sobel)
    Binary[(Scaled_Sobel >= Thresh[0]) & (Scaled_Sobel <= Thresh[1])] = 1    
    
    return Binary
    

def Magnitude_Binary(Img, Sobel_Kernel=3, Thresh=(0, 255)):
    '''
    Returns binary sobel-magnitude image
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
        
    Gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    Sobelx = cv2.Sobel(Gray, cv2.CV_64F, 1, 0, ksize=Sobel_Kernel)
    Sobely = cv2.Sobel(Gray, cv2.CV_64F, 0, 1, ksize=Sobel_Kernel)
    GradMag = np.sqrt(Sobelx**2 + Sobely**2)

    Scale_Factor = np.max(GradMag)/255.0
    GradMag = (GradMag/Scale_Factor).astype(np.uint8)     
    Binary  = np.zeros_like(GradMag)
    Binary[(GradMag >= Thresh[0]) & (GradMag <= Thresh[1])] = 1
    
    return Binary


def Angle_Binary(Img, Sobel_Kernel=3, Thresh=(0, np.pi/2)):
    '''
    Returns binary gradient angle image
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
    
    Gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    
    Sobelx = cv2.Sobel(Gray, cv2.CV_64F, 1, 0, ksize=Sobel_Kernel)
    Sobely = cv2.Sobel(Gray, cv2.CV_64F, 0, 1, ksize=Sobel_Kernel)
    
    AbsGradDir = np.arctan2(np.absolute(Sobely), np.absolute(Sobelx))
    Binary     = np.zeros_like(AbsGradDir)
    
    Binary[(AbsGradDir >= Thresh[0]) & (AbsGradDir <= Thresh[1])] = 1
    
    return Binary


def NOT_Binary(Img):
    '''
    Returns binary image with each pixel flipped
    '''
    Img_NOT = np.zeros_like(Img)
    Img_NOT[Img==0] = 1
    return Img_NOT
    

def AND_Binary(*Imgs):
    '''
    Binary & operation on multiple images
    '''
    Img_AND = np.ones_like(Imgs[0])
    for each in Imgs:
        Img_AND = cv2.bitwise_and(each, Img_AND)
    return Img_AND


def OR_Binary(*Imgs):
    '''
    Binary | operation on multiple images
    '''    
    Img_OR = np.zeros_like(Imgs[0])
    for each in Imgs:
        Img_OR = cv2.bitwise_or(each, Img_OR)
    return Img_OR


def HLS_FromRGB_Binary(Img, Channel, Thresh=(0, 255)):
    '''
    Returns binary H, L, or S image
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
    
    Img_HLS = cv2.cvtColor(Img, cv2.COLOR_RGB2HLS)
    if Channel == 'H':
        Img_Slice = Img_HLS[:,:,0]
    elif Channel == 'L':
        Img_Slice = Img_HLS[:,:,1]
    elif Channel == 'S':
        Img_Slice = Img_HLS[:,:,2]
    else:
        raise NameError('Unexpected HLS channel encountered')
    
    Binary = np.zeros_like(Img_Slice)
    Binary[(Img_Slice > Thresh[0]) & (Img_Slice <= Thresh[1])] = 1
    return Binary


def HSV_FromRGB_Binary(Img, Channel, Thresh=(0, 255)):
    '''
    Returns binary H, S, or V image
    
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
    
    Img_HLS = cv2.cvtColor(Img, cv2.COLOR_RGB2HSV)
    if Channel == 'H':
        Img_Slice = Img_HLS[:,:,0]
    elif Channel == 'S':
        Img_Slice = Img_HLS[:,:,1]
    elif Channel == 'V':
        Img_Slice = Img_HLS[:,:,2]
    else:
        raise NameError('Unexpected HSV channel encountered')
    
    Binary = np.zeros_like(Img_Slice)
    Binary[(Img_Slice > Thresh[0]) & (Img_Slice <= Thresh[1])] = 1
    return Binary


def ImgFilterStack(Binary1, Binary2, SaveFileName=None):
    '''
    Analysis tool to show contributions of each binary filter
    '''
    Color_Binary    = np.dstack((Binary1, np.zeros_like(Binary1), Binary2))*255
    Combined_Binary = np.zeros_like(Binary1)
    Combined_Binary = OR_Binary(Binary1, Binary2)
    
    CompareTwoImages(Color_Binary, Combined_Binary, False, True,
                     'Stacked Image',
                     'Combined Image',
                     SaveFileName)
    

def BinaryVisualization(Binary1, Binary2):
    return np.dstack((Binary1, np.zeros_like(Binary1), Binary2))*255
    

def MagicCorner_ProjectVideoSDC():
    '''
    Hardcoded corners for project video
    '''
    return np.float32([[ 243, 673],
                       [ 589, 446],
                       [ 691, 446],
                       [1037, 673]])  
    
def PlotCorners(Corners):
    '''
    Plots points of Corners array
    '''
    plt.plot(Corners[0][0], Corners[0][1], 'x')
    plt.plot(Corners[1][0], Corners[1][1], 'x')
    plt.plot(Corners[2][0], Corners[2][1], 'x')
    plt.plot(Corners[3][0], Corners[3][1], 'x')
    
    
def Region_of_Interest(Img, Vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    
    Reference: From Udacity's SDC Lane Finding Section
    """    
    Mask = np.zeros_like(Img)   
        
    if len(Img.shape) > 2:
        Channel_Count = Img.shape[2]  # i.e. 3 or 4 depending on image
        Mask_Color = (255,) * Channel_Count
    else:
        Mask_Color = 255
            
    cv2.fillPoly(Mask, Vertices, Mask_Color)
        
    return cv2.bitwise_and(Img, Mask)    


def RectangleCorner(Src, Offset):
    '''
    Maps Src vertices to Dst (Dst is a rectangle)
    '''
    assert(Src[0][1] == Src[3][1]) # expect height of horizontally parallel points to be same
    assert(Src[1][1] == Src[2][1])
    return np.float32([[Src[0][0]+Offset,Src[0][1]],
                       [Src[0][0]+Offset,0],
                       [Src[3][0]-Offset,0],
                       [Src[3][0]-Offset,Src[3][1]],
                      ])

    
def ExpandCorner(Src, Offset, BoundX):
    '''
    Horizontally expand area covered by Src vertices
    Used so that masking region is wide enough to include lane line
    '''
    def CapL(Float):
        return max(BoundX[0], Float)
    def CapR(Float):
        return min(BoundX[1], Float)
    
    return np.float32([[CapL(Src[0][0]-Offset*1.5), Src[0][1]],
                       [     Src[1][0]-Offset   , Src[1][1]],
                       [     Src[2][0]+Offset   , Src[2][1]],
                       [CapR(Src[3][0]+Offset*1.5), Src[3][1]],
                      ])
    
    
def WarpImg(Src, Offset, Img, WarpDirection='>'):
    '''
    Warps Img based on Src and Dst
    '''
    Dst = RectangleCorner(Src, Offset)
    Img_Size = (Img.shape[1], Img.shape[0])
    if WarpDirection == '>':
        M = cv2.getPerspectiveTransform(Src, Dst)
    elif WarpDirection == '<':
        M = cv2.getPerspectiveTransform(Dst, Src)
    else:
        raise NameError('Unexpected warp direction encountered')
    return M, cv2.warpPerspective(Img, M, Img_Size, flags=cv2.INTER_LINEAR)


def LowerHalfHist(Img):
    '''
    Visualization tool - for histogram frequency of lower half of warped image
    '''
    Hist = np.sum(Img[Img.shape[0]//2:,:], axis=0)
    plt.plot(Hist)


def AverageOfBimodal_QuartileTechnique(Hist):
    '''
    Finds midpoint of bimodal peak by taking average of quantile 25, 75 of 
    histogram frequency
    Note: Not robust when left and right lane are not equal in pixel density
    '''
    NonZeroValues = Hist.nonzero()[0]
    p25 = np.percentile(NonZeroValues,25)
    p75 = np.percentile(NonZeroValues,75)
    return int(np.average((p25, p75)))


def AverageOfBimodal_GradientTechnique(Hist):
    '''
    Finds midpoint of bimodal peak by using [+1, -1] kernel
    over nonzero index of histogram
    Note: Robust against unequal pixel density between lanes
    '''
    NonZeroValues = Hist.nonzero()[0]
    NonZeroValues.dtype
    Convoluted = np.zeros((NonZeroValues.shape[0]-1,),
                          dtype=NonZeroValues.dtype)
    
    for each in range(len(Convoluted)):
        Convoluted[each] = NonZeroValues[each] - NonZeroValues[each+1]
        
    GradientIdx = np.max(Convoluted)        
    Peak1       = NonZeroValues[GradientIdx]
    Peak2       = NonZeroValues[GradientIdx+1]
    return int(np.average((Peak1, Peak2)))


def GetLeftAndRightLineFit(Warped_Binary, WindowMargin, MinPixel):
    '''
    Conducts search from bottom of image upwards based on histogram frequency
    
    Returns
    1. Polynomial for left and right lines
    2. Array of windows boxing left and right lines
    3. Array of (x, y) points for left and right lines
            
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''
    
    Hist        = np.sum(Warped_Binary[Warped_Binary.shape[0]//2:,:], axis=0)    
    
    # Find the peak of the left and right halves of the histogram    
    Midpoint    = AverageOfBimodal_GradientTechnique(Hist)
    Leftx_Base  = np.argmax(Hist[:Midpoint])
    Rightx_Base = np.argmax(Hist[Midpoint:]) + Midpoint
    
    nWindows = 9                                     # number of sliding windows    
    Window_Height = Warped_Binary.shape[0]//nWindows # Set height of windows
    
    # Identify the x and y positions of all nonzero pixels in the image
    NonZero  = Warped_Binary.nonzero()
    NonZeroY = np.array(NonZero[0]) # Array of row numbers whose Warped_Binary value is nonzero
    NonZeroX = np.array(NonZero[1])
    
    Leftx_Current  = Leftx_Base  # Current positions to be updated for each window
    Rightx_Current = Rightx_Base
    
    # Create empty lists to receive left and right lane pixel indices
    Left_Lane_Inds  = []
    Right_Lane_Inds = []

    Left_Boxes  = []
    Right_Boxes = []

    # Step through the windows one by one
    for Window in range(nWindows):
        # Identify window boundaries in x and y (and right and left)
        Win_y_Low       = int(Warped_Binary.shape[0] - (Window+1)*Window_Height)
        Win_y_High      = int(Warped_Binary.shape[0] -     Window*Window_Height)
        Win_xLeft_Low   = int(Leftx_Current  - WindowMargin)
        Win_xLeft_High  = int(Leftx_Current  + WindowMargin)
        Win_xRight_Low  = int(Rightx_Current - WindowMargin)
        Win_xRight_High = int(Rightx_Current + WindowMargin)       

        Left_Boxes.append([(Win_xLeft_Low,  Win_y_Low),
                           (Win_xLeft_High, Win_y_High)])

        Right_Boxes.append([(Win_xRight_Low,  Win_y_Low),
                            (Win_xRight_High, Win_y_High)])
    
        # Identify the nonzero pixels in x and y within the window
        Good_Left_Inds  = ( (NonZeroY >= Win_y_Low    ) & (NonZeroY < Win_y_High) 
                          & (NonZeroX >= Win_xLeft_Low) & (NonZeroX < Win_xLeft_High)
                          ).nonzero()[0]
        Good_Right_Inds = ( (NonZeroY >= Win_y_Low     ) & (NonZeroY < Win_y_High) 
                          & (NonZeroX >= Win_xRight_Low) & (NonZeroX < Win_xRight_High)
                          ).nonzero()[0]
    
        Left_Lane_Inds.append(Good_Left_Inds) # Append these indices to the lists
        Right_Lane_Inds.append(Good_Right_Inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(Good_Left_Inds) > MinPixel:
            Leftx_Current  = np.int(np.mean(NonZeroX[Good_Left_Inds]))
        if len(Good_Right_Inds) > MinPixel:
            Rightx_Current = np.int(np.mean(NonZeroX[Good_Right_Inds]))

    # Concatenate the arrays of indices
    Left_Lane_Inds  = np.concatenate(Left_Lane_Inds)
    Right_Lane_Inds = np.concatenate(Right_Lane_Inds)
    
    # Extract left and right line pixel positions
    LeftX  = NonZeroX[Left_Lane_Inds]
    LeftY  = NonZeroY[Left_Lane_Inds] 
    RightX = NonZeroX[Right_Lane_Inds]
    RightY = NonZeroY[Right_Lane_Inds] 

    # Fit a second order polynomial to each
    Left_Fit  = np.polyfit(LeftY,  LeftX,  2)
    Right_Fit = np.polyfit(RightY, RightX, 2)    
    
    return Left_Fit, Right_Fit, \
           Left_Boxes, Right_Boxes, \
           np.vstack((LeftY, LeftX)), np.vstack((RightY, RightX))


def DrawRectangle(Img, Boxes):
    '''
    Draw sliding window boxes on image
    '''
    GREEN    = (0,255,0)
    nWindows = len(Boxes)

    for Window in range(nWindows):        
        cv2.rectangle(Img, 
                      Boxes[Window][0],
                      Boxes[Window][1],
                      GREEN, 3)


def FitPolyLine(LineFit, Rows):
    '''
    Returns array of x and y for ploting quadratic lines
    '''
    Range_Y       = np.linspace(0, Rows-1, Rows)
    F_of_Range_Y  = LineFit[0]*Range_Y**2  + LineFit[1]*Range_Y  + LineFit[2]
    return Range_Y, F_of_Range_Y


def PlotPolyLine(Img, LineFit, LineFit2, LeftBoxes=None, RightBoxes=None, SaveFileName=None):
    '''
    Plots left and right line on Img, with option of line searching boxes
    '''
    if len(Img.shape) < 3:        
        Img = np.zeros((*Img.shape[0:2], 3), dtype=np.uint8)
    
    Rows = Img.shape[0]
    Range_Y, F_of_Range_Y = FitPolyLine(LineFit, Rows)
    _, F2_of_Range_Y      = FitPolyLine(LineFit2, Rows)
    
    if LeftBoxes is not None:        
        DrawRectangle(Img, LeftBoxes)
    if RightBoxes is not None:
        DrawRectangle(Img, RightBoxes)
        
    plt.imshow(Img)
    plt.title('Searched Lines with Boxing Windows')
    plt.plot(F_of_Range_Y,  Range_Y, color='yellow')
    plt.plot(F2_of_Range_Y, Range_Y, color='yellow')
    
    if SaveFileName is not None:        
        plt.savefig(SaveFileName)
         
  
def CreateImgFromPts(LeftFit, RightFit, LeftPts, RightPts, Img_Size):
    '''
    Returns image, red for left lane, blue for right lane
    '''
    RED    = (255,0,0)
    BLUE   = (0,0,255)
    
    Color_Binary = np.zeros((Img_Size[1], Img_Size[0], 3), dtype=np.uint8)
    Color_Binary[LeftPts[0], LeftPts[1]]   = RED
    Color_Binary[RightPts[0], RightPts[1]] = BLUE
        
    return Color_Binary


def GetRadCurvature(LineFit, Rows, Y_pixel_m=30/720.0, X_pixel_m=3.7/700.0):
    '''
    Returns radius of curvature
    '''    
    Range_Y, F_of_Range_Y = FitPolyLine(LineFit, Rows)      
    CURVATURE_POSITION    = np.max(Range_Y) # bottom of image
    
    Fit_Cr  = np.polyfit(Range_Y*Y_pixel_m, F_of_Range_Y*X_pixel_m, 2)
    Curve_Rad = ((1 
                  + (2*Fit_Cr[0]*CURVATURE_POSITION*Y_pixel_m + Fit_Cr[1]
                    )**2
                 )**1.5) / np.absolute(2*Fit_Cr[0])
    return Curve_Rad


def GetOffCentre(LaneCentre, CAR_CENTRE,
                 X_pixel_m=3.7/700.0):
    '''
    Returns distance of car centre to lane centre
    '''
    return np.absolute(LaneCentre-CAR_CENTRE)*X_pixel_m


def FillAreaBetweenLanes(Img,
                         Warped_Binary, Range_Y, 
                         F_of_Range_Y, F2_of_Range_Y,
                         Minv
                         ):
    '''
    Fills area between lanes of warped image and return unwarped image
        
    Reference: Udacity's SDC Advanced Lane Finding code sample
    '''    
    Img_Size   = (Warped_Binary.shape[1], Warped_Binary.shape[0])    
    Img_Zero   = np.zeros_like(Warped_Binary).astype(np.uint8)
    Color_Warp = np.dstack((Img_Zero, Img_Zero, Img_Zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([F_of_Range_Y, Range_Y]))])
    pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([F2_of_Range_Y, Range_Y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(Color_Warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(Color_Warp, Minv, Img_Size)
    
    # Combine the result with the original image
    return cv2.addWeighted(Img, 1, newwarp, 0.3, 0)    


def PutText(Img, Str, Location=(430,670), Font=cv2.FONT_HERSHEY_SIMPLEX):
    '''
    Add text on image frame
    '''
    BLUE = (0,0,255)
    return cv2.putText(Img, Str, Location, Font, 1, BLUE, 2, cv2.LINE_AA)


def DStackSingleChannelImg(Img):    
    return np.dstack((Img, Img, Img))


def P4Pipeline(Img, Mtx, Dist, P4_CORNER):

    Img          = UndistortImg(Img, Mtx, Dist)
    Img_Size     = (Img.shape[1], Img.shape[0])
    
    # Sobel X filter
    THRESH_X     = (12, 255)
    KERNEL_X     = 3    
    Img_Sobelx   = Sobel_Binary(Img, 'x', KERNEL_X, Thresh=THRESH_X)
    
    # Sobel Y filter
    THRESH_Y     = (25, 255)
    KERNEL_Y     = 3    
    Img_Sobely   = Sobel_Binary(Img, 'y', KERNEL_Y, Thresh=THRESH_Y)
    
    # S of HLS space filter
    THRESH_HLS   = (100, 255)
    CHANNEL_HLS  = 'S'
    Img_HLS      = HLS_FromRGB_Binary(Img, CHANNEL_HLS, Thresh=THRESH_HLS)

    # V of HSV space filter
    THRESH_HSV   = (50, 255)
    CHANNEL_HSV  = 'V'
    Img_HSV      = HSV_FromRGB_Binary(Img, CHANNEL_HSV, Thresh=THRESH_HLS)
    
    # Combining filters
    Img_Sobel    = AND_Binary(Img_Sobelx, Img_Sobely)
    Img_HSV_HLS  = AND_Binary(Img_HLS, Img_HSV)
    Img_Combined = OR_Binary(Img_Sobel, Img_HSV_HLS)
    
    Img_Diag1    = BinaryVisualization(Img_Sobel, Img_HSV_HLS)
    
    # Mask image based on region of interest
    BOUNDX       = (150, 1180) # lower left and right mask window cap
    EXPAND_PIXEL = 30 # expand top window by 100, lower window by 100*1.5
    Corner_Expanded = ExpandCorner(P4_CORNER, EXPAND_PIXEL, BOUNDX)
    Img_Masked = Region_of_Interest(Img_Combined,
                                    [Corner_Expanded.astype(np.int32)])
    
    Img_Diag2   = DStackSingleChannelImg(Img_Masked)*255
    
    # Wrap image
    M, Img_Warped = WarpImg(P4_CORNER, 200, Img_Masked, '>')
    Minv, _       = WarpImg(P4_CORNER, 200, Img_Warped, '<')    
    
    _, Img_Diag3 = WarpImg(P4_CORNER, 200, Img, '>')
    
    # Line search using histogram frequency in series of vertical boxes
    # Search continues upwards with centre updating through each layer
    # Also returns fitted line polynomial coefficients
    WINDOW_MARGIN = 80
    MIN_PIXEL     = 100

    LeftFit, RightFit, \
    LeftBoxes, RightBoxes, \
    LeftPts, RightPts \
    = GetLeftAndRightLineFit(Img_Warped, WINDOW_MARGIN, MIN_PIXEL)    
    
    Img_Diag4 = CreateImgFromPts(LeftFit, RightFit, LeftPts, RightPts, Img_Size)
    Img_Diag4 = cv2.addWeighted(Img_Diag3, 1, Img_Diag4, 1, 0)
    
    # Return arrays of x and y points used to plot left, right lines
    Range_Y, F_of_Range_Y_Left = FitPolyLine(LeftFit,  Img_Size[1])
    _, F_of_Range_Y_Right      = FitPolyLine(RightFit, Img_Size[1])

    # Fill area between left and right lanes
    Img_Final = FillAreaBetweenLanes(Img,
                                     Img_Warped,
                                     Range_Y, 
                                     F_of_Range_Y_Left,
                                     F_of_Range_Y_Right,
                                     Minv)

    # Find radius of curvature
    LeftRadCurve  = GetRadCurvature(LeftFit,  Img_Size[1])
    RightRadCurve = GetRadCurvature(RightFit, Img_Size[1])
    AveCurvature  = np.average((LeftRadCurve, RightRadCurve))

    # Find distance between car centre and lane centre
    LeftBaseX  = F_of_Range_Y_Left[-1]
    RightBaseX = F_of_Range_Y_Right[-1]
    LaneCentre = np.average((LeftBaseX, RightBaseX))
    CAR_CENTRE = Img_Size[0]/2 # assume camera is mounted middle of car    
    OffCentre  = GetOffCentre(LaneCentre, CAR_CENTRE)

    # Adds radius of curvature, and distance off centre to image
    Str1 = 'Curvatuve radius: {0:7.2f}m'.format(AveCurvature)
    PutText(Img_Final, Str1)
    
    Str2 = 'Distance off centre: {0:5.2f}m'.format(OffCentre)
    PutText(Img_Final, Str2, Location=(430,640))
        
    return Img_Final, Img_Diag1, Img_Diag2, Img_Diag4


def PipeLineWrapper(f, *args):
    '''
    Hides all parameters of pipeline except Img
    '''
    def _f(Img):        
        return f(Img, *args)
    return _f