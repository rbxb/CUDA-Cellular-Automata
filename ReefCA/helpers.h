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

#include "contants.h"

namespace ReefCA {
	template<typename T = unsigned char>
	void save_pam(std::string name, T* buf, T max = -1, int width = WIDTH, int height = HEIGHT, int depth = DEPTH);

	std::string pad_image_index(int i, int n = 4);
};

#include "helpers.cpp"

#endif // HELPERS_H