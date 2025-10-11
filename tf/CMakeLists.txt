cmake_minimum_required(VERSION 3.10)
project(TensorRTInference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)

# TensorRT paths (adjust these paths based on your TensorRT installation)
set(TensorRT_ROOT /usr/src/tensorrt)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARIES 
    ${TensorRT_ROOT}/lib/libnvinfer.so
    ${TensorRT_ROOT}/lib/libnvonnxparser.so
    ${TensorRT_ROOT}/lib/libnvinfer_plugin.so
)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

# Add executable
add_executable(tensorrt_inference tensorrt_inference.cpp)

# Link libraries
target_link_libraries(tensorrt_inference
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    cudart
    cublas
    curand
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# CUDA architecture (adjust based on your GPU)
set_property(TARGET tensorrt_inference PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Print configuration info
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TensorRT_INCLUDE_DIRS: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT_LIBRARIES: ${TensorRT_LIBRARIES}")