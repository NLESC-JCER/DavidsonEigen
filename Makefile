

LIB_BENCH = -lbenchmark -lpthread
#MKLROOT = /opt/intel/mkl
#INCS = -DMKL_LP64 -m64 -I${MKLROOT}/include -I/usr/lib/x86_64-linux-gnu/openmpi/include 
LIBS =  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 

INCS_CXXOPTS = -I/home/nicolas/Documents/projects/cxxopts-2.1.1/include
INCS_EIGEN = -I/usr/include/eigen3

benchmark : bm_eigen bm_davidson

bm_eigen: benchmark_eigen.cpp
	${CXX} -O3 -fopenmp -march=native ${INCS_EIGEN} $< -o $@ ${LIB_BENCH}

bm_davidson : benchmark_davidson.cpp DavidsonSolver.o DavidsonOperator.o MatrixFreeOperator.o
	${CXX}  -O3 -fopenmp -march=native ${INCS_EIGEN} DavidsonSolver.o DavidsonOperator.o MatrixFreeOperator.o $< -o $@ ${LIB_BENCH}

main: main.cpp DavidsonSolver.o DavidsonOperator.o MatrixFreeOperator.o
	${CXX}  ${INCS_CXXOPTS} ${INCS_EIGEN} DavidsonSolver.o DavidsonOperator.o MatrixFreeOperator.o $< -o $@

DavidsonSolver.o: DavidsonSolver.cpp
	${CXX} ${INCS_EIGEN} -c $< -o $@

DavidsonOperator.o: DavidsonOperator.cpp
	${CXX} ${INCS_EIGEN} -c $< -o $@

MatrixFreeOperator.o: MatrixFreeOperator.cpp
	${CXX} ${INCS_EIGEN} -c $< -o $@