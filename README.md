# kNN-MPI
### Implementation of kNN Search algorithm written in C using MPI library

How to run:
* Git clone the repository
* Open a terminal in the repository
* Type :  ``` make ```

* To execute and validate sequential implementation :
  ```./test_sequential n_arg d_arg k_arg ```

* To execute and validate synchronous(blocking) implementation :
  ```mpirun -np $number_of_processes ./test_synchronous n_per_process_arg d_arg k_arg```

* To execute and validate asynchronous(non-blocking) implementation :
  ```mpirun -np $number_of_processes ./test_asynchronous n_per_process_arg d_arg k_arg```
