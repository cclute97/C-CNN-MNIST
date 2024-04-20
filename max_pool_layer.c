
#include <stdlib.h>
#include "max_pool_layer.h"
#include "cnn_math.h"

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
    int i, j, k, l, m;

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

    layer->output_x_pixel = (int *) malloc(layer->new_width * layer->new_height * sizeof(int));
    layer->output_y_pixel = (int *) malloc(layer->new_height * layer->new_width * sizeof(int));

    int current_region_index = 0;
    for (i = 0; i < layer->new_height; i++) {
        for (j = 0; j < layer->new_width; j++) {
            for (k = 0; k < layer->pool_size; k++) {
                for (l = 0; l < layer->pool_size; l++) {
                    for (m = 0; m < layer->input_num_filters; m++) {
                        layer->input_volume_regions[current_region_index][k][l][m] = input_volume[i * 2 + k][j * 2 + l][m];
                    }
                }
            }
            layer->output_y_pixel[current_region_index] = i;
            layer->output_x_pixel[current_region_index] = j; 
            current_region_index++;
        }
    }
}

double*** max_pool_forward_pass(struct MaxPoolLayer* layer) {
    int i, j;
    double ***output;
    output = (double ***) malloc(layer->new_height * sizeof(double **));
    for (i = 0; i < layer->new_height; i++) {
        output[i] = (double **) malloc(layer->new_width * sizeof(double *));
        for (j = 0; j < layer->new_width; j++) {
            output[i][j] = (double*) malloc(layer->input_num_filters * sizeof(double));
        }
    }

    int max_val_index[] = {0, 0}; // row - column of winning index of given region

    for (i = 0; i < layer->new_height * layer->new_width; i++) {
        find_region_max((const double ***) layer->input_volume_regions[i], (const unsigned short) layer->pool_size, (const unsigned short) layer->input_num_filters, max_val_index);
        for (j = 0; j < layer->input_num_filters; j++) {
            output[layer->output_y_pixel[i]][layer->output_x_pixel[i]][j] = layer->input_volume_regions[i][max_val_index[0]][max_val_index[1]][j];
        }
    }

    return output;
}

void max_pool_free_regions(struct MaxPoolLayer* layer) {

    int i, j, k;

    for (i = 0; i < layer->new_height * layer->new_width; i++) {
        for (j = 0; j < layer->pool_size; j++) {
            for (k = 0; k < layer->pool_size; k++) {
                free(layer->input_volume_regions[i][j][k]);
            }
            free(layer->input_volume_regions[i][j]);
        }
        free(layer->input_volume_regions[i]);
    }
    free(layer->input_volume_regions);

    free(layer->output_x_pixel);
    free(layer->output_y_pixel);
}