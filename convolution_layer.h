
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
     * image_regions -- nx3x3 array containing image regions, where n is the number of 3x3 regions produced from image
     * output_x_pixel -- array of x positions of pixel in output image
     * output_y_pixel -- array of y positions of pixel in output image 
     * All share same index for an output pixel */

    unsigned char ***image_regions;
    unsigned short image_height;
    unsigned short image_width;
    unsigned short *output_x_pixel;
    unsigned short *output_y_pixel;

    /* Function pointers (Note: * comes before variable name when declaring a pointer variable, after when declaring a pointer type) */

    void (*init_layer_fp)(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short); 
    void (*iterate_regions_fp)(struct ConvolutionLayer*, struct Image*); // Generates all 3x3 image regions with valid padding
    struct Image* (*forward_pass_fp)(struct ConvolutionLayer*, struct Image*); // Performs forward pass on image and returns output image
    void (*free_filters_fp)(struct ConvolutionLayer*);
    void (*free_regions_fp)(struct ConvolutionLayer*);
};

void init_layer(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short);
void iterate_regions(struct ConvolutionLayer*, struct Image*);
struct Image* forward_pass(struct ConvolutionLayer*, struct Image*);
void free_filters(struct ConvolutionLayer*);
void free_regions(struct ConvolutionLayer*);

#endif