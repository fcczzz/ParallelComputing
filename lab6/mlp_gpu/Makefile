all: *.cpp *.cu
	nvcc *.cpp *.cu -o mlp `pkg-config --cflags --libs opencv4` -g 

run: all
	./mlp

.PHONY: all run