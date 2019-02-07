 # Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.0.2)

# Project's name
project(davidson_eigensolver)

# required modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(ExternalProject)

include(CTest)
set( CMAKE_VERBOSE_MAKEFILE on )

# Compile with optimization by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Cmake modules/macros are in a subdirectory to keep this file cleaner
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

# Include function to Load MKL
include(${CMAKE_MODULE_PATH}/FindMKL.cmake)

# Find MKL
find_package(MKL)
if (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIRS})
  set(MKL_INCLUDE ${MKL_INCLUDE_DIRS}/intel64/lp64)
  set(LINEAR_ALGEBRA ${MKL_LIBRARIES} pthread m dl)
  message(STATUS "Using MKL for the linear algebra calculations!")
  message(STATUS "MKL_INCLUDE_DIRS: " ${MKL_INCLUDE_DIRS})
  message(STATUS "MKL_LIBRARIES: " ${MKL_LIBRARIES})
else()
  # search for lapack
  find_package(LAPACK COMPONENTS Fortran REQUIRED)
  # Search for Blas
  find_package(BLAS COMPONENTS Fortran REQUIRED)
  set(LINEAR_ALGEBRA ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  message(STATUS "MKL NOT FOUND! Using default BLAS and LAPACK")

  # Find OpenMP
  find_package(OpenMP REQUIRED)

endif(MKL_FOUND)

# search for Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN Include: " ${EIGEN3_INCLUDE_DIR})

#find_package(cxxopts REQUIRED PATHS /home/nicolas/Documents/projects/cxxopts-2.1.1/)
#include_directories(${CXXOPTS_INCLUDE_DIR})
#message(STATUS "CXXOPTS Include: " ${CXXOPTS_INCLUDE_DIR})

# Testing
option(ENABLE_TEST "enable test set" off)


# # search for Doxygen
# find_package(DOXYGEN COMPONENTS REQUIRED)
# MESSAGE(STATUS "DOXYGEN VERSION: " ${DOXYGEN_VERSION})

add_subdirectory(src)
