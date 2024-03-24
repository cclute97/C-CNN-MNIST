# C-CNN-MNIST

Basic implementation of a convolutional neural network (CNN) in C, used to classify characters from the MNIST dataset.  No external libraries used; low level of abstraction to understand core concepts.  Will be adapted and scaled for different datasets/applications. 

Linear Algebra/Data Science Concepts Utilized: 

1. Vectors/Matrices
2. Sobel Filter
3. Convolution
4. Element-wise multiplication
5. Same-Padding / Value-Padding

How Does It Work?

A convolutional neural network is a neural network that utilizes a hidden layer, known as a convolution layer.  They are typically used for computer vision applications due to the 
high number of pixels that that need to be analyzed.  Convolution is used to transform an input image to an output image that is better formated for identifying distinguishing features (edges).  We start by taking an input image and convolving the image with a filter (sobel filter in this case) to produce an output image.  This consists of the following steps: 

   1. Overlaying the filter over the image at a given location.
   2. Performing element wise multiplication between the corresponding values in the image and the overlayed filter.  
   3. Summing the element-wise products, which becomes the output value for the destination pixel of the output image. 
   4. Repeating this process for all location in the input image.  


Sources: 

https://victorzhou.com/blog/intro-to-neural-networks/
https://victorzhou.com/blog/intro-to-cnns-part-1/
https://victorzhou.com/blog/intro-to-cnns-part-2/


TODO:

-Find a dataset to use
-Find a way to convert each image in dataset using convert_image and store in 3D array
-initialize function pointers in init
-functions for 1D->2D array arithmetic
