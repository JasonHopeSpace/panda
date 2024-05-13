cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_STANDARD 17)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.0/bin/nvcc)
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.0)
project(Panda LANGUAGES CXX)
find_package(CUDAToolkit)

project(panda-cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O3")

add_subdirectory(core)

option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif ()

