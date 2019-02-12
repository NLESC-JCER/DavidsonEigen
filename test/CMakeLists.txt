

find_package(Threads REQUIRED)

set(SOURCES test_davidson.cpp ../src/DavidsonSolver.cpp ../src/DavidsonOperator.cpp ../src/MatrixFreeOperator.cpp)
message (STATUS "SOURCES : "  ${SOURCES})
add_executable(test_davidson ${SOURCES})
add_definitions(-DBOOST_TEST_DYN_LINK)

# Add compiler flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DEIGEN_USE_BLAS -DMKL_LP64 -m64 ${BOOST_CFLAGS_PKG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DMKL_LP64 -m64 ${BOOST_CFLAGS_PKG}")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
target_link_libraries(test_davidson -I${EIGEN3_INCLUDE_DIR} ${CMAKE_THREAD_LIBS_INIT} ${LINEAR_ALGEBRA} ${BOOST_LIBS_PKG})


