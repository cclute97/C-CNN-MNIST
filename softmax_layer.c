#include <stdlib.h>
#include <stdio.h>
#include "softmax_layer.h"
#include "cnn_math.h"

void soft_max_init_layer(struct SoftMaxLayer *layer, int input_length, unsigned short nodes) {

    layer->input_length = input_length;
    layer->nodes = nodes;
    int i;

    layer->weights = (double **) malloc(input_length * sizeof(double *));
    for (i = 0; i < input_length; i++) {
        layer->weights[i] = (double *) malloc(nodes * sizeof(double));
    }

    fill_2D_array_random(layer); // fills layer->weights with random values

    layer->biases = (double *) malloc(nodes * sizeof(double));
    for (i = 0; i < nodes; i++) {
        layer->biases[i] = 0;
    }
}

double* soft_max_forward_pass(struct SoftMaxLayer *layer, double*** input_volume, int d1, int d2, int d3) {
    // Convert input volume 3d array -> 1d array
    double *flattened_input;
    double *totals;
    double *output;

    flattened_input = (double *) malloc(d1 * d2 * d3 * sizeof(double));
    totals = (double *) malloc(d1 * d2 * d3 * sizeof(double));
    output = (double *) malloc(d1 * d2 * d3 * sizeof(double)); // free outside of function

    int i, j, k;
    int current = 0;
    for (i = 0; i < d1; i++) {
        for (j = 0; j < d2; j++) {
            for (k = 0; k < d3; k++) {
                flattened_input[current] = input_volume[i][j][k];
                current++;
            }
        }
    }

    dot_product(layer, flattened_input, totals);
    soft_max_function(layer, totals, output);

    //TEST
    printf("\nTotals: \n");
    for (int i = 0; i < d1 * d2 * d3; i++) {
        printf("%f ", totals[i]);
    }
    //TEST

    free(flattened_input);
    free(totals);

    return output;
}

void soft_max_free_layer(struct SoftMaxLayer *layer) {
    for (int i = 0; i < layer->input_length; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);

    free(layer->biases);
}