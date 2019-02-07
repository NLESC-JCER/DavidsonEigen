

find_package(Threads REQUIRED)

set(SOURCES main.cpp DavidsonSolver.cpp DavidsonOperator.cpp MatrixFreeOperator.cpp)
message (STATUS "SOURCES : "  ${SOURCES})
add_executable(main ${SOURCES})


# Add compiler flags
set(CXXOPTS "/home/nicolas/Documents/projects/cxxopts-2.1.1/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DEIGEN_USE_BLAS -DMKL_LP64 -m64 -I${CXXOPTS}/include")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
target_link_libraries(main Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} ${LINEAR_ALGEBRA})
