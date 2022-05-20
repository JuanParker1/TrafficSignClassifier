# TrafficSignClassifier
Machine learning project for classifying images of traffic signs from the German Traffic Sign dataset (GTSRB).

# Overview
#### Files:
<pre>
GTSRB/:                     The dataset
saved_models/:              Contains saved model definitions and weights
TrafficSignClassifier.py:   Main file that runs the application
Dataset_tools.py:           Contains functions for loading and formatting image data from the dataset
Model_train.py:             For running the training process
Model.py:                   Contains model defintions
</pre>
#### Libraries:
<pre>TensorFlow, Matplotlib, Pillow, NumPy</pre>

# Using the Application
The application can be started by running **TrafficSignClassifier.py**. A window will pull up with an image of a
stop sign at the top, and a bar graph on the lower-half. The displayed image is the current selected image from
the dataset (a stop sign by default). To change this, click on either of the two buttons at the bottom of the
window. The first button lets you select an image class from a dropdown menu. The second button lets you select
an image from the dataset directly.

Below the image is a bar graph showing the model's predictions. On the y-axis is the top-5 classes listed from
most likely (top) to least likely (bottom). On the x-axis is the model's confidence for each classification,
ranging from 0.0 to 1.0.

# Dataset
The dataset used in this project is a reduced version of the GTSRB dataset, which consists of over 50,000 images
(~39,000 labeled), split into Train and Test folders. The reduced dataset consists of ~6,500 images from Train and
all ~12,000 images from Test. The purpose of using the reduced training set is that most of the images are of the
same instance / traffic sign with 30 different resolutions. Thus most of the images are redundant. I decided to
pick the top 5 highest-resolution images of each instance.

#### Components:
There are **43 classes** as described in the dataset_dict.txt file.

The **Test** directory consists of ~12,000 unlabled .ppm images. The **Train** directory contains 43 folders
(corresponding to each class), ranging from '00000' to '00042'. Within each folder there is several houndred images
formatted as '000XX_0002i.ppm' where XX is the image instance, and 2i is the resolution (25: lowest, 29: highest).

[Link to the dataset](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
# Model & Training
#### Model:
The model is a convolutional neural network with two consecutive convolution and pooling layers and a residual connection
to the first pooling layer. In addition the model uses dropout, categorical cross-entropy for the loss, and the Adam
optimizer. The model definition is included in the Model.py file, and the saved model is in the saved_models directory
(called "regular_model").
#### Training:
The model was trained for 50 epochs using a batch-size of 16 and early-stopping. 1/5 of the training data was set
aside for validation. Images were converted to grayscale for this model. Since the dataset is fairly small, and there's
only around ~1,300 unique traffic sign images, I decided to employ data augmentation. Random transformations were applied
to each image; namely rotation, width-shift, height-shift, shearing, and zooming.

When running Model_train.py, a TensorBoard session can be launched in one of the following ways:

(With Tensorboard extension in VS Code):
1) Ctrl + Shift + P  to view commands
2) Select "Launch Tensorboard" >> "Use current working directory" (or select folder where logs is contained)

(Without Tensorboard Extension):
1) Open a separate terminal and run "tensorboard --logdir=~/TrafficSignClassifier/logs"
2) Go to the link provided in your web browser (Ex:  "http://localhost:6006/")

NOTE:  Refresh a couple of times until the graphs show up
