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

Will test the full matrix and matrix-free version of the solver and compare the results against Eigen own methods. More benchmark needed.

## Test
```bash
./main 
===========================
= Davidson (DPR)
===========================

iter	Search Space	Norm
   0	          10	7.33e+00/1e-06
   1	          20	2.52e-04/1e-06
   2	          30	7.79e-05/1e-06
   3	          40	2.66e-05/1e-06
   4	          50	8.22e-06/1e-06
   5	          60	2.52e-06/1e-06
   6	          70	6.38e-07/1e-06

Davidson               : 0.352354 secs
Eigen                  : 0.0994993 secs

      Davidson  	Eigen
#   0 0.9996460 	0.9996460
#   1 2.3998684 	2.3998684
#   2 2.5997432 	2.5997432
#   3 5.5994871 	5.5994870
#   4 6.2969455 	6.2969455

===========================
= Matrix Free Method
= Davidson (DPR)
===========================

iter	Search Space	Norm
   0	          10	1.39e+01/1e-06
   1	          20	6.92e-06/1e-06
   2	          30	7.69e-07/1e-06

Davidson               : 0.236414 secs
Eigen                  : 0.0915296 secs

      Davidson  	Eigen
#   0 1.3112183 	1.3112183
#   1 1.5246386 	1.5246386
#   2 5.6588830 	5.6588829
#   3 8.5667100 	8.5667100
#   4 11.6295020 	11.6295018
```