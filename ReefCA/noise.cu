#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"
#include "helpers.cu"

template<typename T = unsigned char>
__global__
void wave_noise(T* buf) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < SIZE) {
		int x = i % WIDTH;
		int y = i / WIDTH;
		float z = sin(float(x) / float(WIDTH) * 8) * sin(float(y) / float(WIDTH) * 8);
		int chance = 30;
		if (z > 0.7) chance = 0;
		else if (z > 0.65) chance = 50;
		else if (z > 0.5) chance = 70;
		else if (z > 0.45) chance = 50;

		int r = helpers::random(i);
		if (chance > abs(r)) buf[i] = 255;
		else buf[i] = 0;
	}
}