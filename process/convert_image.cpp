#include <iostream>
#include <unistd.h>
#include <sys/shm.h>
#include <opencv2/opencv.hpp>

/* Converts an image into a 2D array of pixel intensity values and attached to shared memory segment passed into pipe*/

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cerr << "Invalid number of args.  Expected: 2, Got: " << argc << "\n";
        exit(1);
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); 

    std::cout << image.rows << " " << image.cols << std::endl; // TEST

    if (image.empty()) {
        std::cerr << "Error: Couldn't open or find the image.\n";
        exit(2);
    }

    int pipe_fd = atoi(argv[2]);
    int shared_mem_ID;
    read(pipe_fd, &shared_mem_ID, sizeof(shared_mem_ID));

    unsigned short const rows = image.rows;
    unsigned short const columns = image.cols;
    unsigned char image_matrix[rows][columns]; 

    unsigned char (*shared_mem_seg_pointer)[columns] = (unsigned char(*)[columns])shmat(shared_mem_ID, NULL, 0);

    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            image_matrix[i][j] = image.at<uchar>(i, j);
        }
    }

    shared_mem_seg_pointer = image_matrix;
    shmdt(shared_mem_seg_pointer);

    exit(0);
}
