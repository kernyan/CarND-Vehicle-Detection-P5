# **Vehicle Detection**

---

**Vehicle Detection Project**

#### Executive Summary:
We employed traditional computer vision techniques to vehicle detection in a video stream. By manually selecting a combination of several image features, we achieved a 99.44% classification accuracy on our testing dataset. We then applied the classifier over entire frames using a sliding-window search technique. Using recurring detection across frames, the high confidence vehicles are labeled using bounding boxes.

Outline of the report:
1. Introduction
2. Feature extraction
3. Classifier training and testing
4. Sliding window search
5. Improving detection robustness
6. Combining with lane finding
7. Summary
8. Appendix

[//]: # (Image References)
[image1]: ./examples/test1.jpg "Test image"
[image2]: ./examples/RGB.png "RGB color space"
[image3]: ./examples/HSV.png "HSV color space"
[image4]: ./examples/HLS.png "HLS color space"
[image5]: ./examples/YCrCb.png "YCrCb color space"
[image6]: ./examples/HogFeature.png "Hog feature on car"
[image7]: ./examples/HogFeature2.png "Hog feature on not car"
[image8]: ./examples/SlidingWindows.png "Sliding window 128x128"
[image9]: ./examples/SlidingWindows2.png "Sliding window 64x64"
[image10]: ./examples/Search_S1.png "Search S1"
[video1]: ./project_video.mp4



#### 1. Introduction

The goal of this project is to implement a pipeline that could correctly detect vehicle locations in a video stream using traditional computer vision (CV) techniques. Traditional CV techniques generally emphasize more on manually selecting combinations of image features, as opposed to newer techniques using neural network layers. In this report, we investigated combinations of Histogram of Oriented Gradients (HOG)[1], spatial binning, and color histogram for feature extraction.

Once we have a high accuracy prediction model, we apply the model to small subset of images (with appropriate resizing) across the entire input video frame. The first sliding-window search `search_windows` partitions the entire image into collection of windows (with some overlapping), and for each of the windows we calculated a HOG representation. We also implemented a modified sliding-window search which reuses precalculated HOG features for speed optimization.

Due to the time series nature of video frames, we could leverage on the fact that vehicle locations in one frame will also appear in the same vicinity in subsequent frames. This observation allowed us to introduce a higher confidence detection process by letting the vehicle locations in each frame 'vote' over a fixed set of consecutive frames, and only those locations with votes above a threshold gets accepted as positive detection. This is achieved using a heatmap `MarkHeatmap` and First in First out class `FIFOBoxes`. We used `draw_labeled_bboxes` to fit tighter bounding box on the thresholded detections.

The final vehicle detection pipeline is combined with lane finding pipeline from an earlier project `process_image`. We provide the final video [with](./output_videos/project_video_out.mp4) and [without](./output_videos/project_video_out_Diag.mp4) additional diagnostic data.

#### 2. Feature extraction

Feature extraction is mainly concerned with deriving some form of representation of the image that best aids the classification task of interest. Different techniques expose different aspects of the image's characteristics. Obtaining a good result relies on choosing the right combination of extraction techniques and their corresponding parameters. The techniques we investigated in this report are:

1. Color Space
2. Histogram of Gradients
3. Histogram of Color
4. Spatial binning

##### 2.1 Color Space

We transformed the image below into varies color spaces. For each transformed image, we created a 3D scatter plot where each points corresponds to its actual color in RGB space. The allows us to visually inspect how the pixels in the original images gets represented in the transformed color space. See `Plot3DColorSpace` function in [utils.py](./utils.py)

RGB image
![alt text][image1]

RGB 3D scatter plot
![alt text][image2]

HSV 3D scatter plot
![alt text][image3]

HLS 3D scatter plot
![alt text][image4]

YCrCb 3D scatter plot
![alt text][image5]

Our cars are black and white and hence we are interested to see which color spaces clusters black and white pixels close together. From the scatter plots above, it looks like HLS color space has both black and white pixels on the end of the G axis. However, this visual inspection of pixels are rather subjective. We defer judgment on color space till we measure our model's prediction performance later.

#### 2.2 Histogram of Gradients

Histogram of Gradients was made popular by Dalal, Triggs[1]. The idea is that we partition an image into subregions and run multiple orientation kernels over the pixels. Similar to Sobel kernel over different directions. This allows us to obtain gradient strength in various orientations. Intuitively, HoG features captures the shape of objects. We find that visualization is extremely helpful in gaining an understanding of hog extraction.

Hog Feature on Car
![alt text][image6]

Hog Feature on Not Car
![alt text][image7]

#### 2.3 Color Histogram

Color histogram extraction technique is based on the assumption that objects of the same kind have similar color distribution. For example, a red car seen from the front and side may have very different shape, however they should both have high intensity in its color space that corresponds to red. We perform Color histogram feature extraction is this report by calculating histogram frequency on all three channels of the image's RGB color space. See 'color_hist' in utils.py

#### 2.4 Spatial Binning

Spatial binning is to use the image's raw pixels directly as feature. This may contain some useful information. To reduce the number of features, we employ some downsampling to the image before calculating its histogram frequency. See `bin_spatial` in utils.py

#### 2.5 Experimentation

The best way to find out for sure is to fiddle with them and see what happens. The table below shows the results of our experimentation on 15 different combinations of techniques and parameters. The approach we took to obtain the most information without enumerating all possible combinations is to tweak only one aspect at time, and build up our model once we ascertain the best choice of a particular aspect. The aspects we tested are color spaces, hog pixel per cell, hog cells per block, color histogram bin number, and spatial bin image size. The table below is most useful when looking within their respective section (*in italics*) as it tells us what factor improves accuracy. See `ExtractAndTrainSVC` function in utils.py

| Parameters         		      |  # Features  | Extract (s)| Fit (s)  | Accuracy (%) |
|:--------------------------------|:------------:| :---------:|:--------:|:------------:|
|  *Color Space aspect*           |              |            |          |              |
|  1 HLS					      | 5292         |  77.29     | 17.14    | 98.8739      |
|  2 RGB					      | 5292         |  74.82     | 22.40    | 97.0721      |
|  3 YCrCb 					      | 5292         |  75.68     | 13.61    | 98.8176      |
|  4 HSV 					      | 5292         |  78.44     | 15.63    | 98.8176      |
|                                 |              |            |          |              |
|                                 |              |            |          |              |
|  *Hog pixel per cell aspect*    |              |            |          |              |
|  4 Hog,    8 pix/cell           | 5292   	     |  77.29     | 17.14    | 98.8739      |
|  5 Hog,   16 pix/cell    	      |  972    	 |  45.40     |  2.37    | 97.2973      |
|                                 |              |            |          |              |
|                                 |              |            |          |              |
|  *Hog cell per block aspect*    |              |            |          |              |
|  6 Hog,    2 cell/block  	      |  5292  	     |  77.29     | 17.14    | 98.8739 	    |
|  7 Hog,    4 cell/block         | 10800  	     |  61.56     | 41.21    | 98.3108      |
|                                 |              |            |          |         	    |
|                                 |              |            |          |              |
|  *Color hist bin size aspect*   |              |            |          |              |
|  8 Color Hist, 16 bins  	   	  |  5340 	     | 88.42      | 16.24    | 99.0428      |
|  9 Color Hist, 32 bins   	      |  5388	     | 99.18      | 16.13    | 99.0991      |
|                                 |              |            |          |         	    |
|                                 |              |            |          |              |
| *Spatial bin image size aspect* |              |            |          |              |
| 10 Spatial Bin, (16,16) size    |  6156        | 88.75      | 14.94    | 99.2117	    |
| 11 Spatial Bin, (32,32) size    |  8460        | 88.50      |  4.45    | 99.4369      |
|                              	  |              |            |          |              |
|                                 |              |            |          |              |
| *Combination of techniques*     |              |            |          |              |
| 12 Hog and Color Hist    	      |  5388        | 87.93      | 15.79    | 99.0991      |
| 13 Hog, Spatial Bin             |  8364        | 79.27      |  6.42    | 99.3243      |
| 14 Color Hist, Spatial Bin      |  3168        | 18.29      | 19.71    | 93.2995      |
| 15 Hog, Color Hist, Spatial Bin |  8460        | 88.50      |  4.45    | 99.4369      |

From the table above, we conclude that

| Aspect             	 |  Best     |
|:-----------------------|:---------:|
| Color Space 			 | HLS		 |
| Hog pixel per cell 	 | 8 		 |
| Hog cell per block 	 | 2 		 |
| Color hist bin size 	 | 32 		 |
| Spatial bin image size | (32,32) 	 |
| Techniques to use 	 | All three |

#### 3. Classifier training and testing

Using all three Hog, Color Histogram, and Spatial Binning, we concatenate the features and apply a normalizing operation so that each features has zero mean and unit variance across all training samples. For classification, we used a Support Vector Machine (SVM). As we managed to achieve 99.44% using SVM, we did not investigate other classifiers. See `PrepareTrainTestData`, `TrainSVC` and `ExtractAndTrainSVC` functions in utils.py

#### 4. Sliding window search
1. Multiple scale window and overlap for improving detection
2. Threshold for reducing false positive
3. HOG Subsampling for efficiency

As vehicles could appear in any location in a given image, the classifier has to be applied to 1) various areas of the image, and 2) various scale so that we can detect vehicles of varying sizes. To achieve this, we used a sliding-window search technique. This approach creates window boxes of a specified size over regions of the image. Here's an example of a sliding-window on our test image. See `slide_window` function in utils.py

Sliding-window using (128,128)
![alt][image8]

Sliding-window using (64,64)
![alt][image9]

While this gives us a satisfactory model, it is inefficient in that we were repeating HOG extraction for each window. We then implemented a modified sliding-window search `SearchImageHogSubsample` by calculating HOG extraction for the entire image once and only lookup subregions of HOG representation for each window. We also used heatmap which are votes from each positive detection windows to allow us to know relative confidence of our predictions. See `SearchImageHogSubsample` function in utils.py

Below are some examples our of vehicle detection using the modified sliding-window search and the SVC classifier we trained earlier. See `P5PipelineSimple` function in utils.py

![alt][image10]


#### 5. Improving detection robustness
1. Threshold parameter
2. Scale size
3. Bounding boxes
4. FIFO

Although we see from the example image above that our classifier correctly detects all instances of vehicle with zero false positive, we implemented several further enhancements to our pipeline. See `P5PipelineRobust` function in utils.py

#### 5.1 Two scale sliding-window

We added one more set of sliding-windows with a different scale. By using two sets of sliding-windows with different scales, we allowed our pipeline to be capable of predicting vehicles of different sizes.

#### 5.2 Heatmap threshold

Given that the values in heatmap indicates relative confidence of our detection. We could apply a threshold such that we only retain detection regions of high confidence. This helps to reduce false positives.

#### 5.3 Recurring detection over time

Due to the time series nature of video frames, we could leverage on the fact that vehicle locations in one frame will also appear in the same vicinity in subsequent frames. This observation allowed us to introduce a higher confidence detection process by letting the vehicle locations in each frame 'vote' over a fixed set of consecutive frames, and only those locations with votes above a threshold gets accepted as positive detection. This is achieved using a heatmap `MarkHeatmap` and First in First out class `FIFOBoxes`. We used `draw_labeled_bboxes` to fit tighter bounding box on the thresholded detections.

Knowing that vehicle locations persist in nearby regions across consecutive video frames, we introduced a First in First out class which accumulates the heatmap from each frames. By inspecting the heatmap values in consecutive video frames, we decided that a FIFO of 10 frames and a heatmap threshold of 20 works well. See `FIFOBoxes` class in utils.py

#### 5.4 Bounding-boxes from heatmap

Of the thresholded heatmap, the detection regions are most likely fuzzy in shape. To draw tight bounding boxes around detections, we find the top left and bottom right pixel positions of each continuous blob of nonzero heatmap values. See `draw_labeled_bboxes` function in utils.py

#### 6. Combining with lane finding

We applied our final pipeline to a [project video](./project_video.mp4). Below are some of the processed videos. 

1. [Vehicle Detection with Lane Finding](./output_videos/project_video_out_with_lane.mp4)
2. [Vehicle Detection with Lane Finding with Diagnostic Information](./output_videos/project_video_out_with_lane_diag.mp4)

See `P4Pipeline` in utilsP4.py and `Combine_P4P5` in P5Pipeline.py

#### 7. Discussions

Manually extracting features is a tedious process requiring lots of experimentation. Contrast this with employing a neural network that could train both the vehicle detection and its arbitrary location in the image using convolution. However, the manual process of feature extraction we employed is useful as it helps us build intuition on why and how image detection works.

The pipeline we built still exhibits false positives here and there. To counter this, we could introduce hard negative mining to improve our classifier. We could also consider an ensemble model which combines our current traditional CV approach with a neural network model.

The pipeline may fail to detect vehicles under poor lighting or heavy rain which adds noise to the video frames. Another concern is that the classifier is trained on rather low resolution images of (64, 64) pixels and relatively low sample count (of about 9000 positive and 9000 negative images).

Speed of processing is also an issue. It takes an average home desktop about 30 minutes to process a video of 50 second. This is far from being fast enough for real-time autonomous vehicle steering purposes.  


#### 9. Appendix

#### 9.1 References

1. Navneet Dalal, Bill Triggs. Histograms of Oriented Gradients for Human Detection. June 2005. URL: [http://ieeexplore.ieee.org/document/1467360/](http://ieeexplore.ieee.org/document/1467360/)

#### 9.2 Related files
1. [utils.py](./utils.py) - Python file with functions used for vehicle detection
2. [utilsP4.py](./utils.py) - Python file with functions used for lane finding
3. [P5Pipeline.py](./P5Pipeline.py) - Python file used to process video stream
4. [Vehicle Detection video](./output_videos/project_video_out_with_lane.mp4)
5. [Vehicle Detection video with diagnostics](./output_videos/project_video_out_with_lane_diag.mp4)
