
#ifndef IMAGE_H
#define IMAGE_H

/* A 2D array that represents an image */

struct Image {
    unsigned short width;
    unsigned short height;
    unsigned char **pixel_data;
    void (*init_image_fp)(struct Image*, char*, unsigned short, unsigned short);
    void (*free_image_fp)(struct Image*);
};

void init_image(struct Image*, char*, unsigned short, unsigned short); // call another process (C++/opencv) to build array from image and pass back data to fill our struct
void free_image(struct Image*); // clear up memory

#endif