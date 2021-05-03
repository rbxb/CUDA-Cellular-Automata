#include <chrono>
#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"
#include "helpers.cu"
#include "conway.cu"

long long render(unsigned char* buf_r, unsigned char* buf_w, int frames) {
    // Run seed kernel
    helpers::seed << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Start wall clock
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Run game of life kernel
    for (int i = 0; i < frames; i++) {
        conway::transition << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }

    // Wait for device to finish
    cudaDeviceSynchronize();

    // End wall clock
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
}

int main(void) {
    unsigned char* buffer_a;
    unsigned char* buffer_b;

    // Allocate buffers
    cudaMalloc(&buffer_a, SIZE);
    cudaMalloc(&buffer_b, SIZE);

    std::ofstream ofs;
    ofs.open("frames.csv", std::ios::binary);
    int frames = 2;
    for (int i = 0; i < 20; i++) {
        long long elapsed = render(buffer_a, buffer_b, frames);
        ofs << frames << "," << elapsed << std::endl;
        std::cout << i << "     " << frames << "," << elapsed << std::endl;
        frames *= 2;
    }
    ofs.close();

    // Free buffers
    cudaFree(buffer_a);
    cudaFree(buffer_b);

    return 0;
}