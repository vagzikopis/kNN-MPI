# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
EXECS = test_sequential test_synchronous test_asynchronous

.PHONY: $(EXECS)
all:$(EXECS)
	
test_sequential:
	# tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lopenblas -O3


test_synchronous:
	# tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester_synchronous.c knnring_synchronous.a -o test_synchronous -lm -lopenblas -O3



test_asynchronous:
	# tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester_asynchronous.c knnring_asynchronous.a -o $@ -lm -lopenblas -O3
