/*
 * conway.cuh
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef CONWAY_CUH
#define CONWAY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ReefCA {
	template<int width, int height, int depth, typename T = unsigned char>
	__global__ void conway_transition(T* buf_r, T* buf_w);
};

#include "conway.cu"

#endif // CONWAY_CUH