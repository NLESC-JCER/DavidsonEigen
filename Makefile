

MKLROOT = /opt/intel/mkl
INCS = -DMKL_LP64 -m64 -I${MKLROOT}/include -I/usr/lib/x86_64-linux-gnu/openmpi/include 
LIBS =  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 

INCS_CXXOPTS = -I/home/nicolas/Documents/projects/cxxopts-2.1.1/include
INCS_EIGEN = -I/usr/include/eigen3


main: main.cpp DavidsonSolver.o DavidsonOperator.o
	${CXX}  ${INCS_CXXOPTS} ${INCS_EIGEN} DavidsonSolver.o DavidsonOperator.o $< -o $@

DavidsonSolver.o: DavidsonSolver.cpp
	${CXX} ${INCS_EIGEN} -c $< -o $@

DavidsonOperator.o: DavidsonOperator.cpp
	${CXX} ${INCS_EIGEN} -c $< -o $@