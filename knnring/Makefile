# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
.PHONY: lib

lib:
	cd src; $(CC) -c -O3 -lopenmp knnring_sequential.c; cd ..
	cd src; $(MPICC) -c -O3 -lopenmp knnring_synchronous.c; cd ..
	cd src; $(MPICC) -c -O3 -lopenmp knnring_asynchronous.c; cd ..
	cd src; ar rcs ../lib/knnring_sequential.a knnring_sequential.o; cd ..
	cd src; ar rcs ../lib/knnring_synchronous.a knnring_synchronous.o; cd ..
	cd src; ar rcs ../lib/knnring_asynchronous.a knnring_asynchronous.o; cd ..
