/*
 * noise.cu
 *
 * https://github.com/rbxb/ReefCA
 */

#include "noise.cuh"

#define RAND_SEED 8675309

using namespace ReefCA;

template<typename T>
__device__ T ReefCA::random(int i, T step) {
	i = i % (RAND_SEED * i % 1000);
	unsigned long int n = (i * RAND_SEED) / (RAND_SEED % i);
	unsigned long int range = unsigned long int(T(-1) / step) + 1;
	return T(n % range) * step;
}

template<typename T>
__global__ void ReefCA::seed(T* buf, T step, int width, int height, int depth) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		buf[i * depth] = random(i, step);
	}
}

template<typename T>
__global__ void ReefCA::seed_symmetric(T * buf, char axis, T step, int width, int height, int depth) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		int x = i % width;
		int y = i / width;
		if (axis & MIRROR_X_AXIS == MIRROR_X_AXIS && x >= width / 2) {
			x = width / 2 - (x - width / 2);
		}
		if (axis & MIRROR_Y_AXIS == MIRROR_Y_AXIS && y >= height / 2) {
			y = height / 2 - (y - height / 2);
		}
		int q = y * width + x;
		buf[i * depth] = random(i, step);
	}
}

template<typename T>
__global__ void ReefCA::seed_wave(T* buf, T step, int wave_m, int width, int height, int depth) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width * height) {
		int x = i % width;
		int y = i / width;
		float z = sin(float(x) / float(width) * wave_m) * sin(float(y) / float(width) * wave_m);
		int chance = 60;
		if (z > 0.7f) chance = 0;
		else if (z > 0.67f) chance = 80;
		else if (z > 0.6f) chance = 100;
		else if (z > 0.5f) chance = 80;
		int r = random(i, unsigned char(1)) % 100;
		if (r < chance) buf[i * depth] = random(i, step);
		else buf[i * depth] = 0;
	}
}