
#include <stdio.h>
#include <stdlib.h>
#include "convolution_layer.h"
#include "max_pool_layer.h"
#include "softmax_layer.h"
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

void test_sum_3d_to_1d() {
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

    double *result_1d;
    result_1d = (double *) malloc(layer_01.num_filters * sizeof(double));

    multiply_3d_by_2d(&layer_01, (const unsigned char **) layer_01.image_regions[0], result);
    sum_3d_to_1d(&layer_01, (const double ***) result, result_1d);

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

    printf("Sum of each (filter * region) matrix: \n");
    for (i = 0; i < layer_01.num_filters; i++) {
        printf("%f ", result_1d[i]);
    }
    printf("\n");

    // Free  memory of result
    for (i = 0; i < layer_01.filter_height; i++) {
        for (j = 0; j < layer_01.filter_width; j++) {
            free(result[i][j]);
        }
        free(result[i]);
    }
    free(result); 

    free(result_1d);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
}

void test_forward_pass() {
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
    layer_01.forward_pass_fp = forward_pass;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    double ***output_volume;
    output_volume = layer_01.forward_pass_fp(&layer_01);

    int i, j, k;

    printf("Output Volume: \n");
    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_01.num_filters; k++) {
                printf("%f\n", output_volume[i][j][k]);
            }
            printf("\n");
        }
    }

    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            free(output_volume[i][j]);
        }
        free(output_volume[i]);
    }
    free(output_volume);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
}

void test_forward_pass_large() {
    struct Image image_01;
    image_01.width = 6;
    image_01.height = 6;
    unsigned char data[] = {0, 50, 0, 29, 11, 16, 
                            0, 80, 31, 2, 14, 20,
                            33, 90, 0, 75, 22, 45, 
                            0, 9, 0, 95, 76, 23, 
                            22, 45, 67, 2, 55, 3, 
                            12, 45, 72, 34, 22, 11,
                            11, 60, 2, 4, 66, 72};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.forward_pass_fp = forward_pass;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    double ***output_volume;
    output_volume = layer_01.forward_pass_fp(&layer_01);

    int i, j, k;

    printf("Output Volume: \n");
    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_01.num_filters; k++) {
                printf("%f\n", output_volume[i][j][k]);
            }
            printf("\n");
        }
    }

    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            free(output_volume[i][j]);
        }
        free(output_volume[i]);
    }
    free(output_volume);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
}

void test_max_pool_iterate_regions() {
    struct Image image_01;
    image_01.width = 6;
    image_01.height = 6;
    unsigned char data[] = {0, 50, 0, 29, 11, 16, 
                            0, 80, 31, 2, 14, 20,
                            33, 90, 0, 75, 22, 45, 
                            0, 9, 0, 95, 76, 23, 
                            22, 45, 67, 2, 55, 3, 
                            12, 45, 72, 34, 22, 11,
                            11, 60, 2, 4, 66, 72};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.forward_pass_fp = forward_pass;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    double ***output_volume;
    output_volume = layer_01.forward_pass_fp(&layer_01);

    int i, j, k, l;

    printf("Convolution Layer Output Volume: \n");
    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_01.num_filters; k++) {
                printf("%f\n", output_volume[i][j][k]);
            }
            printf("\n");
        }
    }

    struct MaxPoolLayer layer_02;
    layer_02.init_layer_fp = max_pool_init_layer;
    layer_02.iterate_regions_fp = max_pool_iterate_regions;
    layer_02.init_layer_fp(&layer_02, 2, 4, 4, 10);
    layer_02.iterate_regions_fp(&layer_02, (const double ***) output_volume);

    printf("Max Pooling Layer Regions:\n");
    for (i = 0; i < layer_02.new_height * layer_02.new_width; i++) {
        for (j = 0; j < layer_02.pool_size; j++) {
            for (k = 0; k < layer_02.pool_size; k++) {
                printf("Region %d, row %d, columns %d\n", i, j, k);
                for (l = 0; l < layer_02.input_num_filters; l++) {
                    printf("%f ", layer_02.input_volume_regions[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }

    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            free(output_volume[i][j]);
        }
        free(output_volume[i]);
    }
    free(output_volume);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);

}

void test_max_pool_forward_pass() {
    struct Image image_01;
    image_01.width = 6;
    image_01.height = 6;
    unsigned char data[] = {0, 50, 0, 29, 11, 16, 
                            0, 80, 31, 2, 14, 20,
                            33, 90, 0, 75, 22, 45, 
                            0, 9, 0, 95, 76, 23, 
                            22, 45, 67, 2, 55, 3, 
                            12, 45, 72, 34, 22, 11,
                            11, 60, 2, 4, 66, 72};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.forward_pass_fp = forward_pass;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    double ***output_volume;
    output_volume = layer_01.forward_pass_fp(&layer_01);

    int i, j, k, l;

    printf("Convolution Layer Output Volume: \n");
    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_01.num_filters; k++) {
                printf("%f\n", output_volume[i][j][k]);
            }
            printf("\n");
        }
    }

    struct MaxPoolLayer layer_02;
    layer_02.init_layer_fp = max_pool_init_layer;
    layer_02.iterate_regions_fp = max_pool_iterate_regions;
    layer_02.forward_pass_fp = max_pool_forward_pass;
    layer_02.init_layer_fp(&layer_02, 2, 4, 4, 10);
    layer_02.iterate_regions_fp(&layer_02, (const double ***) output_volume);

    double ***output_volume_2;
    output_volume_2 = layer_02.forward_pass_fp(&layer_02);

    printf("Max Pooling Layer Regions:\n");
    for (i = 0; i < layer_02.new_height * layer_02.new_width; i++) {
        for (j = 0; j < layer_02.pool_size; j++) {
            for (k = 0; k < layer_02.pool_size; k++) {
                printf("Region %d, row %d, columns %d\n", i, j, k);
                for (l = 0; l < layer_02.input_num_filters; l++) {
                    printf("%f ", layer_02.input_volume_regions[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }

    printf("\nMax Pooling Layer Output Volume: \n");
    for (i = 0; i < layer_02.new_height; i++) {
        for (j = 0; j < layer_02.new_width; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_02.input_num_filters; k++) {
                printf("%f ", output_volume_2[i][j][k]);
            }
            printf("\n");
        }
    }

    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            free(output_volume[i][j]);
        }
        free(output_volume[i]);
    }
    free(output_volume);

    for (i = 0; i < layer_02.new_height; i++) {
        for (j = 0; j < layer_02.new_width; j++) {
            free(output_volume_2[i][j]);
        }
        free(output_volume_2[i]);
    }
    free(output_volume_2);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
    layer_02.free_regions_fp = max_pool_free_regions;
    layer_02.free_regions_fp(&layer_02);
}

void test_soft_max_init() {
    struct SoftMaxLayer layer_03;
    layer_03.init_layer_fp = soft_max_init_layer;
    layer_03.init_layer_fp(&layer_03, 2 * 2 * 10, 10);
    int i, j;

    // Print our weights (random nums)
    printf("Weights: (40 x 10)\n");
    for (i = 0; i < (2 * 2 * 10); i++) {
        printf("Row=%d:\n", i);
        for (j = 0; j < 10; j++) {
            printf("%f ", layer_03.weights[i][j]);
        }
        printf("\n");
    }

    // Print biases (all zeros)
    printf("Biases (10x1):\n");
    for (i = 0; i < 10; i++) {
        printf("%f ", layer_03.biases[i]);
    }
    printf("\n");
    
    layer_03.free_layer_fp = soft_max_free_layer;
    layer_03.free_layer_fp(&layer_03);
}

void test_soft_max_forward() {
    struct Image image_01;
    image_01.width = 6;
    image_01.height = 6;
    unsigned char data[] = {0, 50, 0, 29, 11, 16, 
                            0, 80, 31, 2, 14, 20,
                            33, 90, 0, 75, 22, 45, 
                            0, 9, 0, 95, 76, 23, 
                            22, 45, 67, 2, 55, 3, 
                            12, 45, 72, 34, 22, 11,
                            11, 60, 2, 4, 66, 72};

    image_01.pixel_data = data;

    struct ConvolutionLayer layer_01;
    layer_01.init_layer_fp = init_layer;
    layer_01.iterate_regions_fp = iterate_regions;
    layer_01.forward_pass_fp = forward_pass;
    layer_01.init_layer_fp(&layer_01, 10, 3, 3);
    layer_01.iterate_regions_fp(&layer_01, &image_01);

    double ***output_volume;
    output_volume = layer_01.forward_pass_fp(&layer_01);

    int i, j, k, l;

    printf("Convolution Layer Output Volume: \n");
    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_01.num_filters; k++) {
                printf("%f\n", output_volume[i][j][k]);
            }
            printf("\n");
        }
    }

    struct MaxPoolLayer layer_02;
    layer_02.init_layer_fp = max_pool_init_layer;
    layer_02.iterate_regions_fp = max_pool_iterate_regions;
    layer_02.forward_pass_fp = max_pool_forward_pass;
    layer_02.init_layer_fp(&layer_02, 2, 4, 4, 10);
    layer_02.iterate_regions_fp(&layer_02, (const double ***) output_volume);

    double ***output_volume_2;
    output_volume_2 = layer_02.forward_pass_fp(&layer_02);

    printf("Max Pooling Layer Regions:\n");
    for (i = 0; i < layer_02.new_height * layer_02.new_width; i++) {
        for (j = 0; j < layer_02.pool_size; j++) {
            for (k = 0; k < layer_02.pool_size; k++) {
                printf("Region %d, row %d, columns %d\n", i, j, k);
                for (l = 0; l < layer_02.input_num_filters; l++) {
                    printf("%f ", layer_02.input_volume_regions[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }

    printf("\nMax Pooling Layer Output Volume: \n");
    for (i = 0; i < layer_02.new_height; i++) {
        for (j = 0; j < layer_02.new_width; j++) {
            printf("Row: %d, Column: %d \n", i, j);
            for (k = 0; k < layer_02.input_num_filters; k++) {
                printf("%f ", output_volume_2[i][j][k]);
            }
            printf("\n");
        }
    }

    struct SoftMaxLayer layer_03;
    layer_03.init_layer_fp = soft_max_init_layer;
    layer_03.forward_pass_fp = soft_max_forward_pass;
    layer_03.init_layer_fp(&layer_03, 2 * 2 * 10, 10); // layer, dimensions, nodes
    double* output_volume_3 = layer_03.forward_pass_fp(&layer_03, output_volume_2, 2, 2, 10);

    printf("Weights:\n");
    for (i = 0; i < layer_03.input_length; i++) {
        printf("Row=%d\n", i);
        for (j = 0; j < layer_03.nodes; j++) {
            printf("%f ", layer_03.weights[i][j]);
        }
        printf("\n");
    }

    printf("\nSoft Max Layer Output Volume:\n");
    for (i = 0; i < layer_03.input_length; i++) {
        printf("%f ", output_volume_3[i]);
    } 
    printf("\n");

    for (i = 0; i < layer_01.image_height - 2; i++) {
        for (j = 0; j < layer_01.image_width - 2; j++) {
            free(output_volume[i][j]);
        }
        free(output_volume[i]);
    }
    free(output_volume);

    for (i = 0; i < layer_02.new_height; i++) {
        for (j = 0; j < layer_02.new_width; j++) {
            free(output_volume_2[i][j]);
        }
        free(output_volume_2[i]);
    }
    free(output_volume_2);

    free(output_volume_3);

    layer_01.free_regions_fp = free_regions;
    layer_01.free_filters_fp = free_filters;
    layer_01.free_regions_fp(&layer_01);
    layer_01.free_filters_fp(&layer_01);
    layer_02.free_regions_fp = max_pool_free_regions;
    layer_02.free_regions_fp(&layer_02);
    layer_03.free_layer_fp = soft_max_free_layer;
    layer_03.free_layer_fp(&layer_03);
}

int main() {

    //test_layer_init();
    //test_image_init();
    //test_iterate_regions_small();
    //test_iterate_regions_large();
    //test_multiply_3d_by_2d();
    //test_sum_3d_to_1d();
    //test_forward_pass();
    //test_forward_pass_large();
    //test_max_pool_iterate_regions();
    //test_max_pool_forward_pass();
    //test_soft_max_init();
    //test_soft_max_forward();

    return 0;
}