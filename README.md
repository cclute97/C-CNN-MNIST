# C-CNN-MNIST

Basic implementation of a convolutional neural network (CNN) in C, used to classify characters from the MNIST dataset.  No external libraries used; low level of abstraction to understand core concepts.  Will be adapted and scaled for different datasets/applications once the basic implementation is complete. This is currently a long term, ongoing project. Using (mostly) standard library C to basically reinvent numpy is very time consuming.  

Linear Algebra/Data Science Concepts Utilized: 

1. Vectors/Matrices
2. Sobel Filter (?) - Check on this
3. Convolution/Cross-Correlation
4. Element-wise multiplication
5. Value-Padding

A convolutional neural network is a neural network that utilizes a hidden layer, known as a convolution layer.  They are typically used for computer vision applications due to the high number of pixels that need to be analyze. Convolution is used to transform an input image to an output volume that is better formated for identifying distinguishing features (edges).

The convolutional neural network consists of three layers: 

Convolution Layer: We start by taking an input image and convolving the image with a filter to produce an output volume.  The size of our output volume depends on the number of filters used, and the type of padding used.  In this implementation, we will be using valid padding, which produces an output with a height and width smaller than our input by 2 units (ex: 10x10 -> 8x8).  Valid padding provides us with (height - 2 * width - 2) number of regions from our image, with each region having the same dimension as our filters.  Each of those regions is then element-wise multiplied by each of our filters.  Therefore, if our filter_size is 3, our image is 10x10, and our num_filters is 6, then our output volume would be 8x8x6.  

Max Pooling Layer: After we have our output volume from the convolution layer, we pass through a max pool layer.  Because neighboring pixels in images often have similar values, the outut volume from a convolution layer will also have similar values for neighboring pixels.  This is redundant information.  The max pooling layer fixes this by further reducing the size of our output volume by pooling our pixel values.  Max pooling does this by taking a (pool_size x pool_size) region of our output volume from the convolution layer, say a 2x2 region, and taking the highest of the four values, dropping the others.  This yields a new output volume with dimensions (height / 2 x weight / 2 x num_filters). Therefore, if we have an output volume of 8x8x6 from our convolution layer, our output from the max pooling layer would be 4x4x6.  

Softmax Layer: This layer uses the softmax function, which is a useful tool in machine learning used to turn arbitrary values into probabilities (TODO: verify this).  The function does the following: 

   1. Raise e to the power of each given number.
   2. Sum all of the values.  This will be the denominator. 
   3. Each number's exponential is the numerator. 
   4. The probability is numerator/denominator. 

Each output from the softmax function is within the range [0, 1].  These numbers form a probability distribution.  The purpose of performing this operation is to measure how sure we are of our prediction by utilizing cross-entropy loss.  Cross entropy loss is calculated as:
   
L = -ln(Pc) 

c is the correct digit, Pc is the predicted probability for c, and ln is the natural log.  Lower loss is better. 




Sources: 

https://victorzhou.com/blog/intro-to-neural-networks/
https://victorzhou.com/blog/intro-to-cnns-part-1/
https://victorzhou.com/blog/intro-to-cnns-part-2/


TODO:

-Find a dataset to use
-Find a way to convert each image in dataset using convert_image and store in 3D array
-make everything const unless mutable