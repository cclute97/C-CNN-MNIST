
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

void fill_2D_array_random(struct SoftMaxLayer *layer) {
    const double MAX_VALUE = 2147483647.0;
    unsigned short i, j;
    int random_num;
    time_t t;

    srand((unsigned) time(&t));

    for (i = 0; i < layer->input_length; i++) {
        for (j = 0; j < layer->nodes; j++) {
            random_num = rand();
            layer->weights[i][j] = (2 * ((double) random_num / MAX_VALUE) - 1) / layer->input_length;
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

void find_region_max(const double ***region, const unsigned short pool_size, const unsigned short num_filters,
                     int *winning_index) {
    double current_max = region[0][0][0];
    int i, j, k;

    for (i = 0; i < pool_size; i++) {
        for (j = 0; j < pool_size; j++) {
            for (k = 0; k < num_filters; k++) {
                if (region[i][j][k] > current_max) {
                    current_max = region[i][j][k];
                    winning_index[0] = i;
                    winning_index[1] = j;
                }
            }
        }
    }
}   

void dot_product(struct SoftMaxLayer* layer, double *input, double *totals) {
    int i, j;
    double sum = 0;
    for (i = 0; i < layer->input_length; i++) {
        for (j = 0; j < layer->nodes; j++) {
            sum += (layer->weights[i][j] * input[i]) + layer->biases[j];
        }
        totals[i] = sum;
        sum = 0;
    } 
}

void soft_max_function(struct SoftMaxLayer* layer, double *totals, double* output) {
    const double e = 2.71828; // euler's number
    double exp[layer->input_length];
    double sum_exp;
    int i;

    sum_exp = 0;
    for (i = 0; i < layer->input_length; i++) {
        exp[i] = pow(e, totals[i]);
        sum_exp += exp[i];
    }

    for (i = 0; i < layer->input_length; i++) {
        output[i] = exp[i] / sum_exp;
    }
}