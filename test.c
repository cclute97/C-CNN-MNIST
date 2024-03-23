
#include <stdio.h>
#include <stdlib.h>
#include "convolution_layer.h"
#include "image.h"

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

void test_image_init() {
    // Create and initialize image
    struct Image image_01;
    image_01.init_image_fp = init_image; 
    image_01.init_image_fp(&image_01, "process/test.jpg", 463, 703);

    for (int i = 0; i < image_01.width * image_01.height; i++) {
        printf("%d ", image_01.pixel_data[i]);
        if (i % image_01.width == 0) {
            printf("\n");
        }
    }

    image_01.free_image_fp = free_image;
    image_01.free_image_fp(&image_01);
}

int main() {

    //test_layer_init();
    test_image_init();

    return 0;
}