/*
 * cudahelpers.cu
 *
 * https://github.com/rbxb/ReefCA
 */

#include "cudahelpers.cuh"

using namespace ReefCA;

template<int width, int height, int f_depth, int t_depth, typename T>
__host__ __device__ void ReefCA::copy(T* from, T* to) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		to[i * t_depth] = from[i * f_depth];
	}
}

template<int width, int height, int depth>
__host__ __device__ inline int ReefCA::get_rel(int x0, int y0, int x, int y) {
	x = (x0 + x + width) % width;
	y = (y0 + y + height) % height;
	return (y * width + x) * depth;
}

__host__ __device__ inline double ReefCA::dist(int x, int y) {
	return sqrt(x * x + y * y);
}

