CMAKE_MINIMUM_REQUIRED(VERSION 3.18 FATAL_ERROR)

project (DotProduct LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED 11.2)

# adjust this based on your available GPUs
# see https://developer.nvidia.com/cuda-gpus
# 6.1 - 1080Ti
# 7.0 - V100
# 7.5 - 2080Ti
# 8.6 - 3090
set(CMAKE_CUDA_ARCHITECTURES 61 75)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# show kernel details
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  # debug
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
else()
  # profile
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

add_executable(dotProduct testDotProduct.cu)

