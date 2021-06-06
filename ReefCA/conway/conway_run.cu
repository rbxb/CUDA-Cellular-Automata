#include <iostream>
#include <iomanip>

#include "reefca.h"

#define THREADS 256
#define FRAMES 200

#define WIDTH 256
#define HEIGHT 256
#define DEPTH 1

const int SIZE = WIDTH * HEIGHT * DEPTH;

int main(void) {
    unsigned char* buf_r;
    unsigned char* buf_w;

    // Allocate framebuffers
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Run seed kernel
    ReefCA::seed<WIDTH, HEIGHT, DEPTH> << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Loop Conways Game of Life
    for (int i = 0; i < FRAMES; i++) {
        // Copy frame from device to host
        cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);

        // Wait for device to finish
        cudaDeviceSynchronize();

        // Start next transition
        ReefCA::conway_transition<WIDTH, HEIGHT, DEPTH> << < (WIDTH * HEIGHT + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);

        // Update cout
        if (i % 10 == 0) {
            std::cout << i * 100 / FRAMES << "% \t" << i << " of " << FRAMES << std::endl;
        }

        // Save as PAM
        ReefCA::save_pam("out" + ReefCA::pad_image_index(i) + ".pam", out_buffer, WIDTH, HEIGHT, DEPTH);

        // Swap buffers
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }


    // Save the final frame
    cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ReefCA::save_pam("out" + ReefCA::pad_image_index(FRAMES) + ".pam", out_buffer, WIDTH, HEIGHT, DEPTH);

    // Free GPU memory
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::cout << "Done!" << std::endl;

    return 0;
}