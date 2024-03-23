# Makefile for compiling convert_image.cpp

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Libraries
LIBS = `pkg-config --cflags --libs opencv`

# Source file
SRC = convert_image.cpp

# Output executable
TARGET = convert_image

# Makefile rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
