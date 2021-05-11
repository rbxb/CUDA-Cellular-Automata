/* 
 * noise.cuh
 * 
 * https://github.com/rbxb/ReefCA
 */

#ifndef NOISE_CUH
#define NOISE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"

namespace ReefCA {
	const char MIRROR_X_AXIS = 1;
	const char MIRROR_Y_AXIS = 2;

	template<typename T = unsigned char>
	__device__ T random(int i, T step = -1);

	template<typename T = unsigned char>
	__global__ void seed(T* buf, T step = -1, int width = WIDTH, int height = HEIGHT, int depth = DEPTH);

	template<typename T = unsigned char>
	__global__ void seed_symmetric(T* buf, char axis = MIRROR_Y_AXIS, T step = -1, int width = WIDTH, int height = HEIGHT, int depth = DEPTH);

	template<typename T = unsigned char>
	__global__ void seed_wave(T* buf, T step = -1, int wave_m = 8, int width = WIDTH, int height = HEIGHT, int depth = DEPTH);
};

#include "noise.cu"

#endif // NOISE_CUH