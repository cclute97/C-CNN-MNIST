
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
    unsigned char pixel;

    int num_regions = (image->height - 2) * (image->width - 2); // Num of regions when using valid padding

    // Allocate memory for layer->image_region (3D array)
    layer->image_regions = (unsigned char***)malloc(num_regions * sizeof(unsigned char**));
    for (i = 0; i < num_regions; i++) {
        layer->image_regions[i] = (unsigned char**)malloc(layer->filter_height * sizeof(unsigned char*));
        for (j = 0; j < layer->filter_height; j++) {
            layer->image_regions[i][j] = (unsigned char*)malloc(layer->filter_width * sizeof(unsigned char));
        }
    }

    // Allocate memory for layer->output_x_pixel && layer->output_y_pixel
    layer->output_x_pixel = malloc(image->width * image->height * sizeof(unsigned short));
    layer->output_y_pixel = malloc(image->width * image->height * sizeof(unsigned short));

    // Extract all 3x3 sections of image and store
    k = 0;

    for (i = 1; i < image->height - 1; i++) {
        for (j = 1; j < image->width - 1; j++) {
            r = 0;
            for (x = i - 1; x <= i + 1; x++) {
                c = 0;
                for (y = j - 1; y <= j + 1; y++) {
                    pixel = get_element_at(image, (unsigned short) x, (unsigned short) y);
                    layer->image_regions[k][r][c] = pixel; 
                    c++;
                }
                r++;
            }
            layer->output_x_pixel[k] = j;
            layer->output_y_pixel[k] = i;
            k++;
        }
    }
    layer->num_regions = k;
}

// Performs forward pass of the conv layer. Returns 3D array of dimensions:
// (image_height - 2, image_width - 2 , num filters)
unsigned char*** forward_pass(struct ConvolutionLayer* layer) {
    unsigned char ***output; // resulting output volume to return 
    double ***result; // stores result of multiply_3d_by_2d()
    int i, j;

    output = (unsigned char ***) malloc((layer->image_height - 2) * sizeof(unsigned char **));
    for (i = 0; i < layer->image_height - 2; i++) {
        output[i] = (unsigned char **) malloc((layer->image_width - 2) * sizeof(unsigned char *));
        for (j = 0; j < layer->num_filters; j++) {
            output[i][j] = (unsigned char *) malloc(layer->num_filters * sizeof(unsigned char));
        }
    }

    result = (double***) malloc(layer->num_filters * sizeof(double**));    
    for (i = 0; i < layer->num_filters; i++) {
        result[i] = (double**) malloc(layer->filter_height * sizeof(double*));
        for (j = 0; j < layer->filter_height; j++) {
            result[i][j] = (double*) malloc(layer->filter_width * sizeof(double));
        }
    }


    // Free  memory of result
    for (i = 0; i < layer->filter_height; i++) {
        for (j = 0; j < layer->filter_width; j++) {
            free(result[i][j]);
        }
        free(result[i]);
    }
    free(result); 

    return output;
}


// Free Memory Functions // 

void free_filters(struct ConvolutionLayer* layer) {
    int i, j;
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

    free(layer->output_x_pixel);
    free(layer->output_y_pixel);
}