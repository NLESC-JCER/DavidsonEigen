# DavidsonEigen

## Dependencies

Eigen : http://eigen.tuxfamily.org/index.php?title=Main_Page
cxxopts : https://github.com/jarro2783/cxxopts

## Installation
Assuming that eigen is at `/usr/include/eigen`, simply use the Makefile

`
make
`

If Eigen is installed elsewhere change the Makefile accordingly
`
INCS_EIGEN = -I/path/to/eigen/
`

## Usage
`
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
`

Will test the full matrix and matrix-free version of the solver and compare the results against Eigen own methods. More benchmark needed.