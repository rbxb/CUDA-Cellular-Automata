/*
 * mnca.cuh
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef MNCA_CUH
#define MNCA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"
#include <vector>

namespace ReefCA {
	struct nhood {
		int* p;
		int size;
	};

	template<typename T = unsigned char>
	__device__ unsigned long int sum_nhood(T* buf, int x, int y, nhood nh, T threshold = 0);

	template<typename T = unsigned char>
	__global__ void mnca_2n_8t(T* buf_r, T* buf_w, nhood nh0, nhood nh1, unsigned short int* params);

	template<typename T = unsigned char>
	__global__ void draw_nhood(T* buf, int x, int y, nhood nh);

	nhood upload_nh(std::vector<int>& v);

	void generate_nh_fill_circle(int r_outer, int r_inner, std::vector<int>& v);
};

#include "mnca_new.cu"

#endif // MNCA_CUH