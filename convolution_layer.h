
#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "image.h"

struct ConvolutionLayer {

    /* For Initial Creation of Convolution Layer */

    unsigned short filter_width;
    unsigned short filter_height;
    unsigned short num_filters;
    double ***filters; 

    /* For Generation of 3x3 regions of image (using valid padding)
     * image_region -- nx3x3 array containing image regions, where n is the number of 3x3 regions produced from image
     * output_x_pixel -- array of x value's of pixel in output image
     * output_y_pixel -- array of y value's of pixel in output image 
     * All share same index for an output pixel */

    unsigned short ***image_region;
    unsigned short *output_x_pixel;
    unsigned short *output_y_pixel;

    /* Function pointers (Note: * comes before variable name when declaring a pointer variable, after when declaring a pointer type) */

    void (*init_layer_fp)(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short); 
    void (*free_layer_fp)(struct ConvolutionLayer*);
    void (*iterate_regions_fp)(struct Image*);
};

void init_layer(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short);
void free_layer(struct ConvolutionLayer*);
void iterate_regions(struct Image*);

#endif