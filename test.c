
#include <stdio.h>
#include <stdlib.h>
#include "convolution_layer.h"

void test_layer_init() {
    // Create and initialize a layer
    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);

    unsigned short i, j, k;
    
    for (i = 0; i < layer_01.num_filters; i++) {
        for (j = 0; j < layer_01.filter_height; j++) {
            for (k = 0; k < layer_01.filter_width; k++) {
                printf("%f ", layer_01.filters[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {

    test_layer_init();

    return 0;
}