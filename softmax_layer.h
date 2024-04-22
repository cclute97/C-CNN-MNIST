
#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

/* Performs softmax function on output volume of previous layer, max pool
 * soft_max_init_layer() takes in dimensions of input volume, int = d1 * d2 * d3, 
 * and the number of nodes for our softmax layer, unsigned short = number of nodes */

struct SoftMaxLayer {
    double **weights;
    double *biases;
    int input_length;
    unsigned short nodes;

    void(*init_layer_fp)(struct SoftMaxLayer*, int, unsigned short);
    double*(*forward_pass_fp)(struct SoftMaxLayer*, double***, int, int, int); 
    void(*free_layer_fp)(struct SoftMaxLayer*);
};

void soft_max_init_layer(struct SoftMaxLayer*, int, unsigned short);
double* soft_max_forward_pass(struct SoftMaxLayer*, double ***, int, int, int); 
void soft_max_free_layer(struct SoftMaxLayer*);

#endif