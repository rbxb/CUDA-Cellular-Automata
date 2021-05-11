/*
 * cudahelpers.cuh
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef CUDAHELPERS_CUH
#define CUDAHELPERS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"

namespace ReefCA {
	template<typename T = unsigned char>
	__host__ __device__ void copy(T* from, T* to, int width = WIDTH, int height = HEIGHT, int f_depth = DEPTH, int t_depth = DEPTH);

	__host__ __device__ int get_rel(int x0, int y0, int x, int y, int width = WIDTH, int height = HEIGHT, int depth = DEPTH);

	__host__ __device__ int get_rel_fast(int x0, int y0, int x, int y);

	__host__ __device__ double dist(int x, int y);
};

#include "cudahelpers.cu"

#endif // CUDAHELPERS_CUH