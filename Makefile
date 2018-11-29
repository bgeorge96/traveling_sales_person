CC=gcc
NVCC=nvcc

.PHONY: all clean

all: parallel-tsp tsp

parallel-tsp: parallel-tsp.cu
	$(NVCC) -o $@ $^

tsp: tsp-serial.c
	$(CC) -o $@ $^

stest:
	./tsp -c 3
	./parallel-tsp -n 2 -c 3

mtest:
	./tsp -c 4
	./parallel-tsp -n 1 -c 4

homework:
	# make tarball of parallel-tsp and mydoc.pdf

# clean:
# $(RM)
# $(RM) {convolve,convolution-hw}.{pdf,tex}
