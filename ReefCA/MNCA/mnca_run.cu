#include <iostream>
#include <iomanip>
#include <vector>

#include "reefca.h"

#define FRAMES 4000
#define SAVE_INTERVAL 10

int main(void) {
    unsigned char* buf_r;
    unsigned char* buf_w;

    // Create MNCA parameters
    unsigned short int params[8] = { 7,12,19,21,10,25,53,133 };
    unsigned short int* d_params;
    cudaMalloc(&d_params, sizeof(params));
    cudaMemcpy(d_params, &params, sizeof(params), cudaMemcpyHostToDevice);

    // Create neighborhood 0
    std::vector<int> v = std::vector<int>();
    ReefCA::generate_nh_fill_circle(3, 2, v);
    ReefCA::generate_nh_fill_circle(1, 0, v);
    nhood nh0 = ReefCA::upload_nh(v);

    // Create neighborhood 1
    v.clear();
    ReefCA::generate_nh_fill_circle(14, 13, v);
    ReefCA::generate_nh_fill_circle(11, 10, v);
    ReefCA::generate_nh_fill_circle(8, 7, v);
    ReefCA::generate_nh_fill_circle(5, 4, v);
    nhood nh1 = ReefCA::upload_nh(v);

    // Allocate buffers
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Run seed kernel
    ReefCA::seed_wave << < (WIDTHxHEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Loop MNCA generations
    for (int i = 0; i < FRAMES; i++) {
        
        if (i % SAVE_INTERVAL == 0) {
            // Copy frame from device to host
            cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);

            // Wait for device to finish
            cudaDeviceSynchronize();
        }

        // Start next transition
        ReefCA::mnca_2n_8t << < (WIDTHxHEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w, nh0, nh1, d_params);

        // Update cout
        if (i % 10 == 0) {
            std::cout << i * 100 / FRAMES << "% \t" << i << " of " << FRAMES << std::endl;
        }

        if (i % SAVE_INTERVAL == 0) {
            // Save as PPM
            ReefCA::save_pam("out" + ReefCA::pad_image_index(i / SAVE_INTERVAL) + ".pam", out_buffer);
        }
        
        // Swap buffers
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }


    // Save the final frame
    cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ReefCA::save_pam("out" + ReefCA::pad_image_index(FRAMES) + ".pam", out_buffer);

    // Free buffers
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::cout << "Done!" << std::endl;

    return 0;
}