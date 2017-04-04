#**Traffic Sign Recognition** 

##Writeup Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./auxiliar_images/histogram_count_signs.jpg "Visualization"
[image2]: ./auxiliar_images/original_image.jpg "Original"
[image3]: ./auxiliar_images/grayscale_image.jpg "Grayscaling"
[image4]: ./new_images/14_Stop.jpg "Traffic Sign 1"
[image5]: ./new_images/23_SlipperyRoad.jpg "Traffic Sign 2"
[image6]: ./new_images/25_RoadWork.jpg "Traffic Sign 3"
[image7]: ./new_images/31_WildAnimalsCrossing.jpg "Traffic Sign 4"
[image8]: ./new_images/7_SpeedLimit100kmh.jpg "Traffic Sign 5"

[image9]: ./auxiliar_images/Top5_14Stop.png "Top 5 Stop"
[image10]: ./auxiliar_images/Top5_23SlipperyRoad.png "Top 5 Slippery Road"
[image11]: ./auxiliar_images/Top5_25RoadWork.png "Top 5 Road Work"
[image12]: ./auxiliar_images/Top5_31WildAnimalsCrossing.png "Top 5 Wild Animals Crossing"
[image13]: ./auxiliar_images/Top5_7SpeedLimit100kmh.png "Top 5 100 km/h"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

The code for Data Set Summary & Exploration is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for each sign classifier. As the histogram shows, the data is not evenly distributed.

![alt text][image1]

###Design and Test a Model Architecture

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because i noticed that there is no sign with the same image and only changing the colors. So, the color is not important to classify the traffic sign. Converting to grayscale, decrease the number of parameters used on the neural network, demanding less processing.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

As a last step, I normalized the image data with values between -0.9 and 0.9 because it's easier to find a good solution and get close to the global minima of the loss.

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I already downloaded files with training and validation set (train.p and valid.p). So, I don't split my training data into training and validation set using the sklearn.model_selection.train_test_split method.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

The fifth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because I want to increase the accuracy of the model. To add more data to the the data set, I doubled the training set (34799 to 69598 examples) by rotating each image between -10 and +10 degrees randomly and apply cv2.GaussianBlur() in all images because this could insert more information about the images on the neural network. I did not rotate more than 10 degree or flip the images because this could lead to another sign classification (for example, a "turn right ahead" sign could turn into a "turn left ahead" sign). 

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| Probability of keeping units is 0.5			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				| Probability of keeping units is 0.5			|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input: 400, Output: 120						|
| RELU					|												|
| Dropout				| Probability of keeping units is 0.5			|
| Fully connected		| Input: 120, Output: 84						|
| RELU					|												|
| Dropout				| Probability of keeping units is 0.5			|
| Fully connected		| Input: 84, Output: 43							|
| Softmax				| etc.        									|
|						|												|
|						|												|
 

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used the AdamOptimizer optimizer that implements the Adaptive Moment Estimation (Adam) algorithm. I used a final batch size of 512, a number of epochs of 50 and the learning rate of 0.001.

My final model results were:
* training set accuracy of 0.983
* validation set accuracy of 0.949 
* test set accuracy of 0.930

The first architecture used on the model was the LeNet (two convolutional and max pooling layers) followed by three fully connected layers. The LeNet archtecture is a good model for classify images.
The first architecture reached a validation accuracy of 0.874 but it was over fitting (Training accuracy of 0.981). So, I decided to add a dropout layer among every layer. The difference between the training accuracy and the validation accuracy has diminished but the overall accuracy also diminished (under fitting). Then, I increase the number of epochs and the batch size to reach the accuracy required for the project.

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image				        |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Road Work      			| Bumpy road   									| 
| Slippery Road   			| Dangerous curve to the left 					|
| Stop						| Stop											|
| 100 km/h					| Yield							 				|
| Wild Animals Crossing		| Wild Animals Crossing      					|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares unfavorably to the accuracy on the test set of 0.930.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a slippery road sign (probability of 1.00000000e+00), but the image contains a dangerous curve to the left sign. The top five soft max probabilities were

![alt text][image10]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Dangerous curve to the left   				| 
| 5.29600892e-16		| Slippery Road 								|
| 0.00000000e+00		| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)							|
| 0.00000000e+00		| Speed limit (50km/h)      					|


For the second image, the model is sure that this is a wild animals crossing sign (probability of 1.00000000e+00), and the image does contain a wild animals crossing sign. The top five soft max probabilities were

![alt text][image12]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.					| Wild animals crossing   						| 
| 0.					| Speed limit (20km/h) 							|
| 0.					| Speed limit (30km/h)							|
| 0.					| Speed limit (50km/h)							|
| 0.					| Speed limit (60km/h)      					|

For the third image, the model is sure that this is a stop sign (probability of 1.00000000e+00), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image9]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Stop   										| 
| 1.23540027e-23		| Yield 										|
| 0.00000000e+00		| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)							|
| 0.00000000e+00		| Speed limit (50km/h)      					|

For the fourth image, the model is sure that this is a bumpy road sign (probability of 1.00000000e+00), but the image does contain a road work sign. The top five soft max probabilities were

![alt text][image11]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Bumpy road   									| 
| 6.28509086e-23		| Road work 									|
| 3.75129958e-36		| Bicycles crossing								|
| 0.00000000e+00		| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)      					|

For the fifth image, the model is not sure that this is a bumpy road sign (probability of 7.66944051e-01), and the image contain a speed limit 100 km/h sign. The top five soft max probabilities were

![alt text][image13]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.66944051e-01		| Yield   										| 
| 2.33055919e-01		| Ahead only 									|
| 1.80378920e-10		| Keep right									|
| 1.82627459e-13		| No passing for vehicles over 3.5 metric tons	|
| 1.24466732e-25		| Speed limit (80km/h)      					|
   