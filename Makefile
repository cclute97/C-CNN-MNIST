CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
LDFLAGS = -lm
SRC = convolution_layer.c cnn_math.c test.c image.c max_pool_layer.c softmax_layer.c
OBJ = $(SRC:.c=.o)
EXECUTABLE = test

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)
