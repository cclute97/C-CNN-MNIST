
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