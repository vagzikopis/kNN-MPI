# kNN-MPI
### Implementation of kNN Search algorithm written in C using MPI library

How to run:
* Git clone the repository
* Open a terminal in the repository
* Type :  ``` make ```

* To execute sequential implementation : ```./test_sequential n d k ```

* To execute synchronous(blocking) implementation : ```mpirun -np $number_of_processes ./test_synchronous n d k```

* To execute asynchronous(non-blocking) implementation : ```mpirun -np $number_of_processes ./test_asynchronous n d k```

