
#ifndef CNN_MATH_H
#define CNN_MATH_H

#include "convolution_layer.h"

// Fills 3D array filters of given struct
void fill_3D_array_random(struct ConvolutionLayer*);

// Returns element-wise multiply of a 3d and 2d array -- multiplies each 2d array of the 3d by given 2d array
// Copies into second given 3d array.  Both 3d array must have same dimensions. 
void multiply_3d_by_2d(const struct ConvolutionLayer*, const unsigned char**, double***);

// Converts a given 3d array into 1d array, which contains the sum of each 2d array in the given 3d array
// Stores in a given 1d array
void sum_3d_to_1d(const struct ConvolutionLayer*, const double***, double*);

void find_region_max(const double ***, const unsigned short, const unsigned short, int *);

#endif