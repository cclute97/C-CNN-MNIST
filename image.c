
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include "image.h"

void init_image(struct Image* image, char* file_name, unsigned short rows, unsigned short columns) {
    pid_t child_id;
    int pipe_fd[2];
    int shared_memory_ID;

    pipe(pipe_fd);
    child_id = fork();

    if (child_id == -1) {
        fprintf(stderr, "Failed to fork process. Terminating.\n");
        exit(1);
    }

    if (child_id == 0) { // if child process:
        char fd_str[8];
        sprintf(fd_str, "%d", pipe_fd[0]);
        execlp("./process/convert_image", "convert_image", file_name, fd_str); // converts image to 2D array and writes to shared memory segment 
        exit(0);
    }

    else { // if Parent: 
        shared_memory_ID = shmget(IPC_PRIVATE, sizeof(int), IPC_CREAT | 0666);
        close(pipe_fd[0]); 
        write(pipe_fd[1], &shared_memory_ID, sizeof(shared_memory_ID)); 
        close(pipe_fd[1]);
        wait(NULL);
    }

    image->width = columns;
    image->height = rows; 
    image->pixel_data = shmat(shared_memory_ID, NULL, 0);

    shmctl(shared_memory_ID, IPC_RMID, NULL); 
}

void free_image(struct Image* image) {
    unsigned short i;

    for (i = 0; i < image->height; i++) {
        free(image->pixel_data[i]);
    }
    free(image->pixel_data);
}