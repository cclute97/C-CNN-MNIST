
#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

struct ConvolutionLayer {
    unsigned short filter_width;
    unsigned short filter_height;
    unsigned short num_filters;
    double ***filters; 
    // Function pointer -- * comes before variable name when declaring a pointer variable, after when declaring a pointer type
    void (*init_layer_fp)(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short); // Function pointer -- * comes before variable name when declaring a pointer variable, after when declaring a pointer type
    void (*free_layer_fp)(struct ConvolutionLayer*);
};

void init_layer(struct ConvolutionLayer*, unsigned short, unsigned short, unsigned short);

void free_layer(struct ConvolutionLayer*);

#endif