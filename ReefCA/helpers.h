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
	// Writes the texture to a PAM image file
	template<typename T = unsigned char>
	void save_pam(std::string name, T* buf, int width, int height, int depth, T max = -1);

	// Pads a number with zeroes
	// e.g. 0004
	std::string pad_image_index(int i, int n = 4);
};

#include "helpers.cpp"

#endif // HELPERS_H