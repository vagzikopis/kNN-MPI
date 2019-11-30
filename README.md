# kNN-MPI
### Implementation of kNN Search algorithm written in C using MPI library

How to run:
* Git clone the repository
* Open a terminal in the repository
* Type : 
  - ``` cd knnring/ ```
  - ``` make ```
  - ``` cd ..```
* To test sequential implementation : ```make test_sequential```

* To test synchronous(blocking) implementation : ```make test_synchronous```

* To test asynchronous(non-blocking) implementation : ```make test_asynchronous```

* To change parameters change variables : ```n(number of points), d(dimensions), k(k-nearest)``` 
