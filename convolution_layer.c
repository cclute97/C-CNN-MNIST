
#include <stdio.h>
#include <stdlib.h>
#include "convolution_layer.h"
#include "cnn_math.h"

void init_layer(struct ConvolutionLayer* layer, unsigned short num_filters, unsigned short width, unsigned short height) {
    layer->num_filters = num_filters;
    layer->filter_height = height;
    layer->filter_width = width;

    // 3D array 'filters' has dimensions (num_filters, height, width) or (depth, row, column)
    // dynamically allocate our array to heap

    unsigned short i;
    unsigned short j;

    layer->filters = (double***)malloc(num_filters * sizeof(double**));    
    for (i = 0; i < num_filters; i++) {
        layer->filters[i] = (double**)malloc(height * sizeof(double*));
        for (j = 0; j < height; j++) {
            layer->filters[i][j] = (double*)malloc(width * sizeof(double));
        }
    }
    fill_3D_array_random(layer); // in 'cnn_math.h'
}

void free_layer(struct ConvolutionLayer* layer) {
    unsigned short i;
    unsigned short j;

    for (i = 0; i < layer->filter_height; i++) {
        for (j = 0; j < layer->filter_width; j++) {
            free(layer->filters[i][j]);
        }
        free(layer->filters[i]);
    }
    free(layer->filters);
}