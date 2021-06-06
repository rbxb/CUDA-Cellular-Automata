/*
 * cudahelpers.cuh
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef CUDAHELPERS_CUH
#define CUDAHELPERS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ReefCA {
	// Copies one channel from one texture to another
	template<int width, int height, int f_depth, int t_depth, typename T = unsigned char>
	__host__ __device__ void copy(T* from, T* to);

	// Gets the index of the pixel relative to the pixel at x0,y0
	template<int width, int height, int depth>
	__host__ __device__ inline int get_rel(int x0, int y0, int x, int y);

	// Distance formula
	__host__ __device__ inline double dist(int x, int y);
};

#include "cudahelpers.cu"

#endif // CUDAHELPERS_CUH