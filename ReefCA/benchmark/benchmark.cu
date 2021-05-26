#include <chrono>
#include <fstream>
#include <iostream>

#include "reefca.h"

#define THREADS 256

#define WIDTH 256
#define HEIGHT 256
#define DEPTH 1

const int SIZE = WIDTH * HEIGHT * DEPTH;

typedef void render_func (unsigned char* buf_r, unsigned char* buf_w);

long long render(render_func rf, unsigned char* buf_r, unsigned char* buf_w, int frames) {
    // Run seed kernel
    ReefCA::seed<WIDTH, HEIGHT, DEPTH> << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Start wall clock
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Run game of life kernel
    for (int i = 0; i < frames; i++) {
        rf(buf_r, buf_w);
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }

    // Wait for device to finish
    cudaDeviceSynchronize();

    // End wall clock
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
}

void benchmark(render_func rf, std::string name, unsigned char* buffer_a, unsigned char* buffer_b) {
    std::ofstream ofs;
    ofs.open(name + ".csv", std::ios::binary);
    ofs << "frames,time" << std::endl;
    int frames = 2;
    for (int i = 0; i < 16; i++) {
        long long elapsed = render(rf, buffer_a, buffer_b, frames);
        ofs << frames << "," << elapsed << std::endl;
        std::cout << i << "     " << frames << "," << elapsed << std::endl;
        frames *= 2;
    }
    ofs.close();
}

void rf_conway_transition(unsigned char* buf_r, unsigned char* buf_w) {
    ReefCA::conway_transition<WIDTH, HEIGHT, DEPTH> << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);
}

int main(void) {
    unsigned char* buffer_a;
    unsigned char* buffer_b;

    // Allocate buffers
    cudaMalloc(&buffer_a, SIZE);
    cudaMalloc(&buffer_b, SIZE);

    benchmark(rf_conway_transition, "benchmark", buffer_a, buffer_b);

    // Free buffers
    cudaFree(buffer_a);
    cudaFree(buffer_b);

    return 0;
}