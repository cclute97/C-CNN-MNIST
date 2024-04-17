
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cnn_math.h"

void fill_3D_array_random(struct ConvolutionLayer* layer) {
    const double MAX_VALUE = 2147483647.0;
    unsigned short i, j, k;
    int random_num;
    time_t t;

    srand((unsigned) time(&t));

    for (i = 0; i < layer->num_filters; i++) {
        for (j = 0; j < layer->filter_height; j++) {
            for (k = 0; k < layer->filter_width; k++) {
                random_num = rand();
                layer->filters[i][j][k] = 2 * ((double) random_num / MAX_VALUE) - 1; // scale down our random number
            }
        }
    }
}

void multiply_3d_by_2d(const struct ConvolutionLayer* layer, const unsigned char** array_2d, double*** result) {
    int i, j, k;
    for (i = 0; i < layer->num_filters; i++) {
        for (j = 0; j < layer->filter_height; j++) {
            for (k = 0; k < layer->filter_width; k++) {
                result[i][j][k] = layer->filters[i][j][k] * array_2d[j][k];
            }
        }
    }
}

void sum_3d_to_1d(const struct ConvolutionLayer *layer, const double*** array_3d, double* array_1d) {
    double sum = 0;
    int i, j, k;
    for (i = 0; i < layer->num_filters; i++) {
        for (j = 0; j < layer->filter_height; j++) {
            for (k = 0; k < layer->filter_width; k++) {
                sum += array_3d[i][j][k];
            }
        }
        array_1d[i] = sum;
        sum = 0;
    }
}