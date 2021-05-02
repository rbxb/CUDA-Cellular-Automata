#ifndef HELPERS_CU
#define HELPERS_CU

#include <math.h>
#include <vector>
#include <string>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"

// Writes the buffer as PAM image
void save_image(std::string path, char* buf, int width, int height, int depth) {
    std::ofstream ofs;
    ofs.open(path, std::ios::binary);
    ofs << "P7" << std::endl;
    ofs << "WIDTH " << width << std::endl;
    ofs << "HEIGHT " << height << std::endl;
    ofs << "DEPTH " << depth << std::endl;
    ofs << "MAXVAL 255" << std::endl;
    ofs << "ENDHDR" << std::endl;
    ofs.write(buf, width * height * depth);
    ofs.close();
}

// Kernel to sum all pixels in the array of indices
__device__
int sum_array(int* a, int size, char* buf, int rel = 0) {
    unsigned int count = 0;
    for (int i = 0; i < size; i++) {
        count += buf[(a[i] + rel) % SIZE];
    }
    return count;
}

// Kernel to get a pseudorandom number from an index and a seed
__device__
int random(int i) {
    return (RAND_SEED * i + RAND_SEED) * (i % RAND_SEED) % 100;
}

// Kernel function to seed cellmap
__global__
void seed(char* buf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE && random(i) > 70) buf[i] = 255;
}

// Kernel function that gets pixel index relative to current pixel
__device__
int get_rel(int x0, int y0, int x, int y) {
    x = (x0 + x + WIDTH) % WIDTH;
    y = (y0 + y + HEIGHT) % HEIGHT;
    return y * WIDTH + x;
}

// Distance function
double dist(int x, int y) {
    return sqrt(x * x + y * y);
}

// Generates an index list for a disk with inner and outer radius
int* generate_nh_radius(int r_outer, int r_inner, int* size) {
    std::vector<int> a = std::vector<int>();
    for (int x = -r_outer; x < r_outer; x++) {
        for (int y = -r_outer; y < r_outer; y++) {
            double d = dist(x, y);
            if (d <= r_outer && d > r_inner) a.push_back(y * WIDTH + x);
        }
    }
    *size = a.size();
    return &a[0];
}

#endif // HELPERS_CU