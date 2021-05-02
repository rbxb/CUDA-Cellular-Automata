# CMakeList.txt : CMake project for ReefCA
cmake_minimum_required (VERSION 3.9)

project(ReefCA LANGUAGES CXX CUDA)

add_executable(benchmark "ReefCA/benchmark/benchmark.cu")
add_executable(conway "ReefCA/conway/conway_run.cu")

target_compile_features(benchmark PUBLIC cxx_std_11)
target_compile_features(conway PUBLIC cxx_std_11)

set_target_properties(benchmark conway PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(benchmark conway PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark conway PROPERTIES CUDA_ARCHITECTURES "35;50;72")

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
include_directories("ReefCA")