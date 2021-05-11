# CMakeList.txt : CMake project for ReefCA
cmake_minimum_required (VERSION 3.9)

project(ReefCA LANGUAGES CXX CUDA)

add_executable(benchmark "ReefCA/benchmark/benchmark.cu")
add_executable(conway "ReefCA/conway/conway_run.cu")
add_executable(mnca_test "ReefCA/mnca/mnca_test.cu")
add_executable(mnca_run "ReefCA/mnca/mnca_run.cu")
#add_executable(mnca_gen "ReefCA/mnca/mnca_gen.cu")

target_compile_features(benchmark PUBLIC cxx_std_11)
target_compile_features(conway PUBLIC cxx_std_11)
target_compile_features(mnca_test PUBLIC cxx_std_11)
target_compile_features(mnca_run PUBLIC cxx_std_11)
#target_compile_features(mnca_gen PUBLIC cxx_std_11)

set_target_properties(conway benchmark mnca_test mnca_run PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(conway benchmark mnca_test mnca_run PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(conway benchmark mnca_test mnca_run PROPERTIES CUDA_ARCHITECTURES "35;50;72")

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
include_directories("ReefCA")