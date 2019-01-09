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
./main --size 1000
Matrix size : 1000x1000
Num Threads : 1

===========================
= Davidson (DPR)
===========================

iter	Search Space	Norm
   0	          10	1.55e+00/1e-06
   1	          20	3.11e-03/1e-06
   2	          30	1.38e-03/1e-06
   3	          40	9.50e-04/1e-06
   4	          50	6.80e-04/1e-06
   5	          60	4.69e-04/1e-06
   6	          70	3.36e-04/1e-06
   7	          80	2.48e-04/1e-06
   8	          90	1.90e-04/1e-06
   9	         100	1.54e-04/1e-06
  10	          10	2.33e-14/1e-06

Davidson               : 14.6725 secs
Eigen                  : 54.108 secs

      Davidson  	Eigen
#   0 1.0952436 	1.0951738
#   1 1.4963254 	1.4962028
#   2 1.5963869 	1.5962247
#   3 1.8952878 	1.8949336
#   4 1.9888033 	1.9884468

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
#   0 0.3994790 	0.3994790
#   1 0.6749251 	0.6749251
#   2 1.0814170 	1.0814170
#   3 1.3745905 	1.3745903
#   4 1.4875875 	1.4875875

```