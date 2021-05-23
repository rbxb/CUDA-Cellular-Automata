#include <chrono>
#include <fstream>
#include <iostream>

#include "reefca.h"

typedef void render_func (unsigned char* buf_r, unsigned char* buf_w);

long long render(render_func rf, unsigned char* buf_r, unsigned char* buf_w, int frames) {
    // Run seed kernel
    ReefCA::seed << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r);

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
    ReefCA::conway_transition << < (WIDTHxHEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);
}

void rf_conway_transition_fast(unsigned char* buf_r, unsigned char* buf_w) {
    ReefCA::conway_transition_fast << < (WIDTHxHEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);
}

unsigned short int* d_params;
ReefCA::nhood nh0, nh1;

void setup_mnca() {
    // Create MNCA parameters
    unsigned short int params[8] = { 7,12,19,21,10,25,53,133 };

    cudaMalloc(&d_params, sizeof(params));
    cudaMemcpy(d_params, &params, sizeof(params), cudaMemcpyHostToDevice);

    // Create neighborhood 0
    std::vector<int> v = std::vector<int>();
    ReefCA::generate_nh_fill_circle(3, 2, v);
    ReefCA::generate_nh_fill_circle(1, 0, v);
    nh0 = ReefCA::upload_nh(v);

    // Create neighborhood 1
    v.clear();
    ReefCA::generate_nh_fill_circle(14, 13, v);
    ReefCA::generate_nh_fill_circle(11, 10, v);
    ReefCA::generate_nh_fill_circle(8, 7, v);
    ReefCA::generate_nh_fill_circle(5, 4, v);
    nh1 = ReefCA::upload_nh(v);
}

void rf_mnca(unsigned char* buf_r, unsigned char* buf_w) {
    ReefCA::mnca_2n_8t << < (WIDTHxHEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w, nh0, nh1, d_params);
}

int main(void) {
    unsigned char* buffer_a;
    unsigned char* buffer_b;

    // Allocate buffers
    cudaMalloc(&buffer_a, SIZE);
    cudaMalloc(&buffer_b, SIZE);

    benchmark(rf_conway_transition, "conway", buffer_a, buffer_b);
    benchmark(rf_conway_transition_fast, "conway_fast", buffer_a, buffer_b);

    setup_mnca();
    benchmark(rf_mnca, "mnca", buffer_a, buffer_b);

    // Free buffers
    cudaFree(buffer_a);
    cudaFree(buffer_b);

    return 0;
}