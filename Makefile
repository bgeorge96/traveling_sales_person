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
	./tsp -c 5
	./parallel-tsp -n 3 -c 5

# clean:
# $(RM)
# $(RM) {convolve,convolution-hw}.{pdf,tex}
