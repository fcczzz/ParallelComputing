all: *.cpp *.cu
	nvcc *.cpp *.cu -o mlp `pkg-config --cflags --libs opencv4` -g -O3

run: all
	./mlp

.PHONY: all run