#ifndef HELPERS_CU
#define HELPERS_CU

#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"

namespace helpers {

    // Writes the buffer as PAM image
    void save_image(std::string path, unsigned char* buf, int width, int height, int depth) {
        std::ofstream ofs;
        ofs.open(path, std::ios::binary);
        ofs << "P7" << std::endl;
        ofs << "WIDTH " << width << std::endl;
        ofs << "HEIGHT " << height << std::endl;
        ofs << "DEPTH " << depth << std::endl;
        ofs << "MAXVAL 255" << std::endl;
        ofs << "TUPLTYPE GRAYSCALE" << std::endl;
        ofs << "ENDHDR" << std::endl;
        ofs.write((char*)buf, width * height * depth);
        ofs.close();
    }

    // Pad image index with zeroes
    std::string pad_image_index(int i, int n = 4) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(n) << i;
        return ss.str();
    }

    // Kernel to get a pseudorandom number from an index and a seed
    __device__
    int random(int i) {
        return (RAND_SEED * i + RAND_SEED) * (i % RAND_SEED) % 100;
    }

    // Kernel function to seed cellmap
    __global__
    void seed(unsigned char* buf) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < SIZE && random(i) > 60) buf[i] = 255;
    }

    // Kernel function to symmetrically seed cellmap
    __global__
        void seed_symmetric(unsigned char* buf) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < SIZE) {
            int x = i % WIDTH;
            int y = i / WIDTH;
            if (x >= WIDTH / 2) {
                x = WIDTH / 2 - (x - WIDTH / 2);
            }
            if (y >= HEIGHT / 2) {
                y = HEIGHT / 2 - (y - HEIGHT / 2);
            }
            int q = y * WIDTH + x;
            if (random(q) > 70) {
                buf[i] = 255;
            }
        }
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
};

#endif // HELPERS_CU