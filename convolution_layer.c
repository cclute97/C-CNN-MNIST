
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    // Allocate memory for layer->filters (3D array)
    layer->filters = (double***)malloc(num_filters * sizeof(double**));    
    for (i = 0; i < num_filters; i++) {
        layer->filters[i] = (double**)malloc(height * sizeof(double*));
        for (j = 0; j < height; j++) {
            layer->filters[i][j] = (double*)malloc(width * sizeof(double));
        }
    }
    fill_3D_array_random(layer); // in 'cnn_math.h'
}

// Generate all possible 3x3 image regions using valid padding -- store in layer 
// Note: valid padding reduces dimensions by 2 each, hence the width/height - 2
void iterate_regions(struct ConvolutionLayer* layer, struct Image* image) {

    int i, j, k, x, y, r, c;
    unsigned char sub_array[3][3];

    // Allocate memory for layer->image_region (3D array)
    layer->image_regions = (unsigned char***)malloc(3 * 3 * ((image->height - 2) * (image->width - 2)) * sizeof(unsigned char**));
    for (i = 0; i < (image->height - 2) * (image->width - 2); i++) {
        layer->image_regions[i] = (unsigned char**)malloc(image->height * sizeof(unsigned char*));
        for (j = 0; j < 3; j++) {
            layer->image_regions[i][j] = (unsigned char*)malloc(image->width * sizeof(unsigned char));
        }
    }

    // Extract all 3x3 sections of image and store
    k = 0;
    r = 0;
    c = 0;

    for (i = 1; i < image->height - 1; i++) {
        for (j = 1; j < image->width - 1; j++) {
            for (x = i - 1; x <= i + 1; x++) {
                c = 0;
                for (y = j - 1; y <= j + 1; y++) {
                    sub_array[r][c] = get_element_at(image, (unsigned short) x, (unsigned short) y);
                    c++;
                }
                r++;
            }
            memcpy(layer->image_regions[k], sub_array, sizeof(sub_array));
            layer->output_x_pixel[k] = j;
            layer->output_y_pixel[k] = i;
            r = 0;
            k++;
        }
    }
}

// Free Memory Functions // 

void free_filters(struct ConvolutionLayer* layer) {
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

void free_regions(struct ConvolutionLayer* layer) {
    unsigned short i;
    unsigned short j;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            free(layer->image_regions[i][j]);
        }
        free(layer->image_regions[i]);
    }
    free(layer->image_regions);
}