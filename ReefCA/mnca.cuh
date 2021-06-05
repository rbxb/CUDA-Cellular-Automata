/*
 * mnca.cuh
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef MNCA_CUH
#define MNCA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

namespace ReefCA {
	struct nhood {
		int* p;
		int size;
	};

	template<typename T = unsigned char>
	struct rule {
		unsigned short int nh;
		unsigned long int lower;
		unsigned long int upper;
		T value;
	};

	template<int width, int height, int depth, typename T = unsigned char>
	__device__ unsigned long int sum_nhood(T* buf, int x, int y, nhood* nh, T threshold = 0);

	template<int width, int height, int depth, typename T = unsigned char, int max_nhs = 2>
	__global__ void mnca_transition(T* buf_r, T* buf_w, nhood* nhs, rule<T>* rules, int n);

	template<int width, int height, int depth, typename T = unsigned char>
	__global__ void draw_nhood(T* buf, int x, int y, nhood nh);

	template<int width, int height, int depth, typename T = unsigned char>
	__global__ void draw_nhood(T* buf, int x, int y, nhood* nh);

	nhood upload_nh(std::vector<int>& v);
	nhood* upload_nh_array(std::vector<nhood>& v);

	void generate_nh_fill_circle(int r_outer, int r_inner, std::vector<int>& v);

	template<typename T = unsigned char>
	bool read_mnca_rule(nhood** nhs, int* num_nhs, rule<T>** rules, int* num_rules);
};

#include "mnca.cu"

#endif // MNCA_CUH