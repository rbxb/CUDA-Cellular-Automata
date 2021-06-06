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
	// Contains the pixel offsets for a neighborhood
	// p is usually a pointer to an array on the GPU with x,y offset values
	// size is the number of neighbors
	struct nhood {
		int* p;
		int size;
	};

	// A rule has a pair of thresholds and a value to set the cell to
	template<typename T = unsigned char>
	struct rule {
		int nh;
		unsigned long int lower;
		unsigned long int upper;
		T value;
	};

	// Sums the neighbors
	// It will only count a neighbor if the value is greater than the threshold
	template<int width, int height, int depth, typename T = unsigned char>
	__device__ unsigned long int sum_nhood(T* buf, int x, int y, nhood* nh, T threshold = 0);

	// A generalized MNCA transition function
	// Based on slackermanz's
	template<int width, int height, int depth, typename T = unsigned char, int max_nhs = 2>
	__global__ void mnca_transition(T* buf_r, T* buf_w, nhood* nhs, rule<T>* rules, int n);

	// Draws a neighborhood centered at x,y
	template<int width, int height, int depth, typename T = unsigned char>
	__global__ void draw_nhood(T* buf, int x, int y, nhood* nh);

	// Uploads a neighborhood to the GPU
	// v is a vector of x,y offsets
	nhood upload_nh(std::vector<int>& v);

	// Fills v with x,y offsets for a disk
	// 
	// NOT TESTED RECENTLY
	void generate_nh_fill_circle(int r_outer, int r_inner, std::vector<int>& v);

	// Reads an MNCA rule from the standard input and uploads the neighborhoods and rules
	template<typename T = unsigned char>
	bool read_mnca_rule(nhood** nhs, int* num_nhs, rule<T>** rules, int* num_rules);

	// Frees GPU memory used by the offset values for the array of nhoods
	// Does not free the nhood array itself
	void free_nhs_values(nhood* nhs, int n);
};

#include "mnca.cu"

#endif // MNCA_CUH