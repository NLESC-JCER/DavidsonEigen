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

## test
`
./main --size 100
`

Will test the full matrix and matrix-free version of the solver and comapre the results against Eigen.
More benchmark needed.