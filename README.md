DavidsonEigen
===================

This package contains a C++ implementation of the *Davidson diagonalization algorithms*. The calculation can be performed *matrix free*, i.e. without having to ever store the entire matrix. Different schemas are available to compute the correction.

Available correction methods are:
 * **DPR**: Diagonal-Preconditioned-Residue
 * **GJD**: Generalized Jacobi Davidson

### Note:
The Davidson method is suitable for **diagonal-dominant symmetric matrices**, that are quite common
in certain scientific problems like [electronic structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson method could be not practical
for other kind of symmetric matrice.

Usage
-----

The following program instantiates the `DavidsonOperator` class and uses the `DavidsonSolver` to compute
the lowest 5 eigenvalues and corresponding eigenvectors, using the *GJD* method.

```C++
#include <iostream>
#include "DavidsonSolver.hpp"
#include "DavidsonOperator.hpp"

int main (int argc, char *argv[]) {

    // Matrix Free Operator
    DavidsonOperator A(1000);

    // Davidson Solver
    DavidsonSolver DS;
    DS.set_jacobi_correction();
    DS.set_jacobi_linsolve("CG");
    DS.solve(A,5);

    auto eigenvalues = DS.eigenvalues();
    auto eigenvectors = DS.eigenvectors();

    return;
}
```

### References:
 * [Davidson diagonalization method and its applications to electronic structure calculations](https://pdfs.semanticscholar.org/57811/eaf768d1a006f505dfe24f329874a679ba59.pdf?_ga=2.219777566.664950272.1547548596-1327556406.1547548596)
 * [Numerical Methods for Large Eigenvalue Problem](https://doi.org/10.1137/1.9781611970739)



Installation
------------------------

To compile execute:
```
cmake -H. -Bbuild && cmake --build build
```

Dependencies
------------
This packages assumes that you have installed the following packages:
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
 * [cxxopts](https://github.com/jarro2783/cxxopts)
 * [Lapack](http://www.netlib.org/lapack/)

Optionally, If an [MKL](https://software.intel.com/en-us/mkl) library is available the package will try to find it.

Binaries
---------

The cmake compiles a binary in `bin/` that can be directly used to test the routines :

```bash
./main --help
Eigen Davidson Iterative Solver
Usage:
  ./main [OPTION...]
      --size arg      dimension of the matrix (default: 100)
      --neigen arg    number of eigenvalues required (default: 5)
      --jocc          use Jacobi-Davidson
      --linsolve arg  method to solve the linear system of JOCC (0:CG,
                      1:GMRES, 2:LLT) (default: 0)
      --help          Print the help
```

```bash
./main --size 1000
Matrix size : 1000x1000
Num Threads : 1

===========================
= Matrix Free Method
= Davidson (DPR)
===========================

iter	Search Space	Norm
   0	          10	9.22e-01/1e-06
   1	          20	4.62e-06/1e-06
   2	          30	2.49e-06/1e-06
   3	          40	1.57e-07/1e-06

Davidson               : 27.6797 secs
Eigen                  : 53.4498 secs

      Davidson  	Eigen
   0 0.3994790 	0.3994790
   1 0.6749251 	0.6749251
   2 1.0814170 	1.0814170
   3 1.3745905 	1.3745903
   4 1.4875875 	1.4875875

```