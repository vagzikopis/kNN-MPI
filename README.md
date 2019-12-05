# kNN-MPI
### Implementation of kNN Search algorithm written in C using MPI library

How to run:
* Git clone the repository
* Open a terminal in the repository
* Type :  ``` make ```

* To execute sequential implementation :
  ```./test_sequential n_arg d_arg k_arg ```

* To execute synchronous(blocking) implementation : 
  ```mpirun -np $number_of_processes ./test_synchronous n_arg d_arg k_arg```

* To execute asynchronous(non-blocking) implementation : 
  ```mpirun -np $number_of_processes ./test_asynchronous n_arg d_arg k_arg```

