
#include <stdio.h>
#include <stdlib.h>
#include "convolution_layer.h"
#include "image.h"
#include "cnn_math.h"

// Create and initialize a layer_01
void test_layer_init() {

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

// Create and initialize image
void test_image_init() {

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

void test_iterate_regions_small() {

    struct Image image_01;
    image_01.width = 4;
    image_01.height = 4;
    unsigned char data[] = {0, 50, 0, 29, 
                            0, 80, 31, 2, 
                            33, 90, 0, 75, 
                            0, 9, 0, 95};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    int i, j, k;

    for (i = 0; i < (layer_01.image_width - 2) * (layer_01.image_height - 2); i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                printf("%d ", layer_01.image_regions[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
}

// Iterate regions of image and create 3D array of various regions stored in layer_01
void test_iterate_regions_large() {

    struct Image image_01;
    image_01.init_image_fp = init_image; 
    image_01.init_image_fp(&image_01, "process/test.jpg", 463, 703);

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    //int i, j, k;
    
    /*
    for (i = 0; i < ((image_01.width - 2) * (image_01.height - 2)); i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                printf("%d ", layer_01.image_regions[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    
    image_01.free_image_fp = free_image;
    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    image_01.free_image_fp(&image_01);
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
}

void test_multiply_3d_by_2d() {
    struct Image image_01;
    image_01.width = 4;
    image_01.height = 4;
    unsigned char data[] = {0, 50, 0, 29, 
                            0, 80, 31, 2, 
                            33, 90, 0, 75, 
                            0, 9, 0, 95};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    int i, j, k;
    double ***result;
    result = (double***) malloc(layer_01.num_filters * sizeof(double**));    
    for (i = 0; i < layer_01.num_filters; i++) {
        result[i] = (double**) malloc(layer_01.filter_height * sizeof(double*));
        for (j = 0; j < layer_01.filter_height; j++) {
            result[i][j] = (double*) malloc(layer_01.filter_width * sizeof(double));
        }
    }

    multiply_3d_by_2d(&layer_01, (const unsigned char **) layer_01.image_regions[0], result);

    printf("Current filters: \n");
    for (i = 0; i < layer_01.num_filters; i++) {
        for (j = 0; j < layer_01.filter_height; j++) {
            for (k = 0; k < layer_01.filter_width; k++) {
                printf("%f ", layer_01.filters[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    unsigned char **region = layer_01.image_regions[0];
    printf("Given image region: \n");
    for (i = 0; i < layer_01.filter_height; i++) {
        for (j = 0; j < layer_01.filter_width; j++) {
            printf("%d ", region[i][j]);
        }
        printf("\n");
    }

    printf("\nCurrent filters mutliplied by given region: \n");
    for (i = 0; i < layer_01.num_filters; i++) {
        for (j = 0; j < layer_01.filter_height; j++) {
            for (k = 0; k < layer_01.filter_width; k++) {
                printf("%f ", result[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free  memory of result
    for (i = 0; i < layer_01.filter_height; i++) {
        for (j = 0; j < layer_01.filter_width; j++) {
            free(result[i][j]);
        }
        free(result[i]);
    }
    free(result); 

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);

}

int main() {

    //test_layer_init();
    //test_image_init();
    //test_iterate_regions_small();
    //test_iterate_regions_large();
    //test_multiply_3d_by_2d();

    return 0;
}