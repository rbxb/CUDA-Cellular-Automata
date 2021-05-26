/*
 * helpers.h
 *
 * https://github.com/rbxb/ReefCA
 */

#ifndef HELPERS_H
#define HELPERS_H

#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ReefCA {
	template<typename T = unsigned char>
	void save_pam(std::string name, T* buf, int width, int height, int depth, T max = -1);

	std::string pad_image_index(int i, int n = 4);
};

#include "helpers.cpp"

#endif // HELPERS_H