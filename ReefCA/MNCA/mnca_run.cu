#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "reefca.h"

#define FRAMES 4000
#define SAVE_INTERVAL 10
#define THREADS 256

#define WIDTH 2048
#define HEIGHT 2048
#define DEPTH 1

const int SIZE = WIDTH * HEIGHT * DEPTH;

int main(void) {
    ReefCA::nhood* nhs;
    ReefCA::rule<unsigned char>* rules;
    int num_nhs;
    int num_rules;
    ReefCA::read_mnca_rule(&nhs, &num_nhs, &rules, &num_rules);

    // Allocate framebuffers
    unsigned char* buf_r;
    unsigned char* buf_w;
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Allocate out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Run seed noise kernel
    ReefCA::seed_wave<WIDTH, HEIGHT, DEPTH> << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Loop MNCA generations
    for (int i = 0; i < FRAMES; i++) {
        
        if (i % SAVE_INTERVAL == 0) {
            // Copy frame from device to host
            cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);

            // Wait for device to finish
            cudaDeviceSynchronize();
        }

        // Start next transition
        ReefCA::mnca_transition<WIDTH, HEIGHT, DEPTH> 
            << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > 
            (buf_r, buf_w, nhs, rules, num_rules);

        // Update cout
        if (i % 10 == 0) {
            std::cout << i * 100 / FRAMES << "% \t" << i << " of " << FRAMES << std::endl;
        }

        if (i % SAVE_INTERVAL == 0) {
            // Save as PPM
            ReefCA::save_pam("out" + ReefCA::pad_image_index(i / SAVE_INTERVAL) + ".pam", out_buffer, WIDTH, HEIGHT, DEPTH);
        }
        
        // Swap buffers
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }


    // Save the final frame
    cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ReefCA::save_pam("out" + ReefCA::pad_image_index(FRAMES) + ".pam", out_buffer, WIDTH, HEIGHT, DEPTH);

    // Free buffers
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::cout << "Done!" << std::endl;

    return 0;
}