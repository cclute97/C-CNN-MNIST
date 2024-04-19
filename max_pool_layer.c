
#include <stdlib.h>
#include "max_pool_layer.h"

void max_pool_init_layer(struct MaxPoolLayer* layer, unsigned char pool_size, const unsigned short input_height, 
                const unsigned short input_width, const unsigned short num_filters) {
    layer->pool_size = pool_size;
    layer->height_before_pooling = input_height;
    layer->width_before_pooling = input_width;
    layer->input_num_filters = num_filters;

    layer->new_height = layer->height_before_pooling / layer->pool_size;
    layer->new_width = layer->width_before_pooling / layer->pool_size;
}

void max_pool_iterate_regions(struct MaxPoolLayer* layer, const double ***input_volume) {
    int i, j, k, l;

    layer->input_volume_regions = (double ****) malloc(layer->new_height * layer->new_width * sizeof(double ***));
    for (i = 0; i < layer->new_height * layer->new_width; i++) {
        layer->input_volume_regions[i] = (double ***) malloc(layer->pool_size * sizeof(double **));
        for(j = 0; j < layer->pool_size; j++) {
            layer->input_volume_regions[i][j] = (double **) malloc(layer->pool_size * sizeof(double *));
            for (k = 0; k < layer->pool_size; k++) {
                layer->input_volume_regions[i][j][k] = (double *) malloc(layer->input_num_filters * sizeof(double));
            }
        }
    }

    layer->output_x_pixel = (int *) malloc(layer->new_width * sizeof(int));
    layer->output_y_pixel = (int *) malloc(layer->new_height * sizeof(int));

    int current_region_index = 0;
    for (i = 0; i < layer->new_height; i++) {
        for (j = 0; j < layer->new_width; j++) {
            for (k = 0; k < layer->pool_size; k++) {
                for (l = 0; l < layer->pool_size; l++) {
                    layer->input_volume_regions[current_region_index][k][l] = (double *) input_volume[i * 2 + k][j * 2 + l];
                }
            }
            layer->output_y_pixel[current_region_index] = i;
            layer->output_x_pixel[current_region_index] = j; 
            current_region_index++;
        }
    }
}