cmake_minimum_required(VERSION 3.8)
SET(CMAKE_CXX_FLAGS "-std=c++11")
project(CUDA_TEST)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(main0 reduce_sum_v0.cu)
cuda_add_executable(main1 reduce_sum_v1.cu)
target_link_libraries(main0 ${CUDA_LIBRARIES})
target_link_libraries(main1 ${CUDA_LIBRARIES})