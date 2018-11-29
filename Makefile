CC=gcc
RM=rm
NVCC=nvcc

.PHONY: all clean

all: parallel-tsp tsp

parallel-tsp: parallel-tsp.cu
	$(NVCC) -o $@ $^

tsp: tsp-serial.c
	$(CC) -o $@ $^

test3:
	./tsp -c 3 -s 56
	./parallel-tsp -n 1 -c 3 -s 56
	./parallel-tsp -n 2 -c 3 -s 56
	./parallel-tsp -n 4 -c 3 -s 56
	./parallel-tsp -n 8 -c 3 -s 56

test4:
	./tsp -c 4 -s 56
	./parallel-tsp -n 1 -c 4 -s 56
	./parallel-tsp -n 2 -c 4 -s 56
	./parallel-tsp -n 4 -c 4 -s 56
	./parallel-tsp -n 8 -c 4 -s 56
	./parallel-tsp -n 16 -c 4 -s 56
	./parallel-tsp -n 32 -c 4 -s 56

test5:
	./tsp -c 5 -s 56
	./parallel-tsp -n 1 -c 5 -s 56
	./parallel-tsp -n 2 -c 5 -s 56
	./parallel-tsp -n 4 -c 5 -s 56
	./parallel-tsp -n 8 -c 5 -s 56
	./parallel-tsp -n 16 -c 5 -s 56
	./parallel-tsp -n 32 -c 5 -s 56
	./parallel-tsp -n 32 -c 5 -s 56
	./parallel-tsp -n 64 -c 5 -s 56
	./parallel-tsp -n 128 -c 5 -s 56

test6:
	./tsp -c 6 -s 56
	./parallel-tsp -n 1 -c 6 -s 56
	./parallel-tsp -n 2 -c 6 -s 56
	./parallel-tsp -n 4 -c 6 -s 56
	./parallel-tsp -n 8 -c 6 -s 56
	./parallel-tsp -n 16 -c 6 -s 56
	./parallel-tsp -n 32 -c 6 -s 56
	./parallel-tsp -n 32 -c 6 -s 56
	./parallel-tsp -n 64 -c 6 -s 56
	./parallel-tsp -n 128 -c 6 -s 56
	./parallel-tsp -n 256 -c 6 -s 56
	./parallel-tsp -n 512 -c 6 -s 56
	./parallel-tsp -n 1024 -c 6 -s 56

test7:
	./tsp -c 7 -s 56
	./parallel-tsp -n 4 -c 7 -s 56
	./parallel-tsp -n 8 -c 7 -s 56
	./parallel-tsp -n 16 -c 7 -s 56
	./parallel-tsp -n 32 -c 7 -s 56
	./parallel-tsp -n 32 -c 7 -s 56
	./parallel-tsp -n 64 -c 7 -s 56
	./parallel-tsp -n 128 -c 7 -s 56
	./parallel-tsp -n 256 -c 7 -s 56
	./parallel-tsp -n 512 -c 7 -s 56
	./parallel-tsp -n 1024 -c 7 -s 56

test8:
	./tsp -c 8 -s 56
	./parallel-tsp -n 8 -c 8 -s 56
	./parallel-tsp -n 16 -c 8 -s 56
	./parallel-tsp -n 32 -c 8 -s 56
	./parallel-tsp -n 32 -c 8 -s 56
	./parallel-tsp -n 64 -c 8 -s 56
	./parallel-tsp -n 128 -c 8 -s 56
	./parallel-tsp -n 256 -c 8 -s 56
	./parallel-tsp -n 512 -c 8 -s 56
	./parallel-tsp -n 1024 -c 8 -s 56

test9:
	./tsp -c 9 -s 56
	./parallel-tsp -n 8 -c 9 -s 56
	./parallel-tsp -n 16 -c 9 -s 56
	./parallel-tsp -n 32 -c 9 -s 56
	./parallel-tsp -n 32 -c 9 -s 56
	./parallel-tsp -n 64 -c 9 -s 56
	./parallel-tsp -n 128 -c 9 -s 56
	./parallel-tsp -n 256 -c 9 -s 56
	./parallel-tsp -n 512 -c 9 -s 56
	./parallel-tsp -n 1024 -c 9 -s 56

test10:
	./tsp -c 10 -s 56
	./parallel-tsp -n 8 -c 10 -s 56
	./parallel-tsp -n 16 -c 10 -s 56
	./parallel-tsp -n 32 -c 10 -s 56
	./parallel-tsp -n 32 -c 10 -s 56
	./parallel-tsp -n 64 -c 10 -s 56
	./parallel-tsp -n 128 -c 10 -s 56
	./parallel-tsp -n 256 -c 10 -s 56
	./parallel-tsp -n 512 -c 10 -s 56
	./parallel-tsp -n 1024 -c 10 -s 56

experiment: test3 test4 test5 test6 test7 test8 test9 test10
	$^

pdf:
	pdflatex mydoc.tex
	rm mydoc.aux mydoc.log

homework: parallel-tsp.cu mydoc.pdf
	tar zcvf $@ $^

clean:
	$(RM) parallel-tsp tsp-serial
