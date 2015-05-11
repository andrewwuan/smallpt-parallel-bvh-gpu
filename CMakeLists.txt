# CMakeLists.txt for G4CU project
project(15418-parallel-raytracing)

# required cmake version
cmake_minimum_required(VERSION 2.8)

#if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#        SET(CMAKE_CXX_COMPILER /usr/local/bin/g++-4.9)
#endif()

# CUDA package
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch compute_20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
cuda_add_executable(smallpt smallpt.cu gpu.cu)
