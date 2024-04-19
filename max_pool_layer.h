#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H

/* Performs max pooling of output volume from convolution layer.  Treats output of convolution layer, output_volume,
 * as input, input_volume.  Produces a new, smaller output volume as the result of the forward pass. 
 
 * iterate_regions() takes our input_volume and creates a new array, input_volume_regions, that contains all 
 * valid non-overlaping pool_size x pool_size regions of our input_volume. 
  
 * forward_pass() takes each region generated from iterate_regions(), determines which array at each region index
 * contains the largest value, and then maps that array to the current output pixel. 
 */

struct MaxPoolLayer {
    unsigned char pool_size;
    unsigned short height_before_pooling;
    unsigned short width_before_pooling;
    unsigned short new_height;
    unsigned short new_width;
    unsigned short input_num_filters;
    int *output_x_pixel;
    int *output_y_pixel;
    double ****input_volume_regions; // array of dimensions -- (new_width * new_height) x pool_size x pool_size x num_filters

    void(*init_layer_fp)(struct MaxPoolLayer*, unsigned char, const unsigned short, const unsigned short,
                         const unsigned short);
    void(*iterate_regions_fp)(struct MaxPoolLayer*, const double ***);
    void(*forward_pass_fp)(struct MaxPoolLayer*, const double ***);
    void(*free_regions_fp)(struct MaxPoolLayer*);
};

void max_pool_init_layer(struct MaxPoolLayer*, unsigned char, const unsigned short, const unsigned short,
                const unsigned short);
void max_pool_iterate_regions(struct MaxPoolLayer*, const double ***);
void max_pool_forward_pass(struct MaxPoolLayer*, const double ***);
void max_pool_free_regions(struct MaxPoolLayer*);

#endif 