# C-CNN-MNIST

Basic implementation of a convolutional neural network (CNN) in C, used to classify characters from the MNIST dataset.  No external libraries used; low level of abstraction to understand core concepts.  

Linear Algebra Concepts Utilized: 

1. Vectors/Matrices
2.

Explanation of CNN Concepts:

1. The basic building block of a neural network is a neuron.  A neuron takes inputs, performs some mathematical operations on them, and produces one output.  
2. Connections between neurons have a weight.  They determine the strength of the connection and how much influence one neuron's ouput has on another neuron's
   input.  They can be seen as coefficients that adjust the impact of incoming data.  They can increase or decrease the importance of specific information. 
   They are how the neural network learns from data.  
3. Each neuron has a corresponding bias, which is a constant associated with that neuron.  They are not connected so specific input, but are added to the output 
   of a neuron.  They serve as a form of threshold or offset.  Allows for flexibility and identifying more complex patterns.  They are parameters that can be fine
   tuned to improve the performance of a neural network.  
4. A neural network consists of neurons strung together.  There is an input layer, output layer, and various hidden layers betwen the input and ouput layers.  
   A neural network can have infinitely many layers and neurons.  
5. Forward propagation is the initial phase of processing data through a neural network to produce output.  
6. Back propagation is where the output is evaluated for accuracy to determine which adjustments need to be made, and updates weights and biases.  
7. "loss" is how you quantify how "good" a neural network is doing, so that it can try to do better.  Ex. 'Mean Squared Error Loss' formula - takes the average
   over all squared errors.  The better the predictions are, the lower the loss is.  Training a network = minimizing loss. 
8. Optimization algorithm 'stochastic gradient descent' is used to tell the neural network how to change weights and biases to minimize loss. 
9. A convolutional neural network is just as neural network with a convolution layer.  These layers consist of a set of filters, which are matrices of numbers. Ex. 3x3 filter known
   as Sobel filter, which emphasizes edges (vertical or horizontal edges depending on if vertical or horizontal filter is used).  Convolution helps us look for specific localized image features (like edges).  Primary parameter of this layer is the number of filters it has. 

Sources: 

https://victorzhou.com/blog/intro-to-neural-networks/
https://victorzhou.com/blog/intro-to-cnns-part-1/
https://victorzhou.com/blog/intro-to-cnns-part-2/
https://www.geeksforgeeks.org/the-role-of-weights-and-bias-in-neural-networks/


 