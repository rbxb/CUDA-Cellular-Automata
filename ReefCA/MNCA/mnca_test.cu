#include <iostream>
#include <iomanip>

#include "reefca.h"

#define WIDTH 256
#define HEIGHT 256
#define DEPTH 1

const int SIZE = WIDTH * HEIGHT * DEPTH;

int main(void) {
    ReefCA::nhood* nhs;
    ReefCA::rule<unsigned char>* rules;
    int num_nhs;
    int num_rules;
    ReefCA::read_mnca_rule(&nhs, &num_nhs, &rules, &num_rules);

    // Allocate buffers
    unsigned char* buf_w;
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Draw neighborhood
    ReefCA::draw_nhood<WIDTH, HEIGHT, DEPTH> << < 1, 1 >> > (buf_w, 20, 20, nhs);

    // Copy frame from device to host
    cudaMemcpy(out_buffer, buf_w, SIZE, cudaMemcpyDeviceToHost);

    // Wait for device to finish
    cudaDeviceSynchronize();

    // Save as PPM
    ReefCA::save_pam("mnca_test.pam", out_buffer, WIDTH, HEIGHT, DEPTH);
    
    // Free buffers
    cudaFree(buf_w);

    return 0;
}