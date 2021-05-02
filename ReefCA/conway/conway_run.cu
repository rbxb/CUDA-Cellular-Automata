#include <iostream>
#include <iomanip>

#include "conway.cu"
#include "helpers.cu"

#define FRAMES 100

int main(void) {
    char* buf_r;
    char* buf_w;

    // Allocate buffers
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    char* out_buffer = new char[SIZE];

    // Run seed kernel
    seed << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Loop conways game of life
    for (int i = 0; i < FRAMES; i++) {
        // Copy frame from device to host
        cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);

        // Wait for device to finish
        cudaDeviceSynchronize();

        // Start next transition
        transition << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r, buf_w);

        // Update cout
        if (i % 10 == 0) {
            std::cout << i * 100 / FRAMES << "% \t" << i << " of " << FRAMES << std::endl;
        }

        // Save as PPM
        save_image("out-" + std::to_string(i) + ".pam", out_buffer, WIDTH, HEIGHT, 1);

        // Swap buffers
        char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }


    // Save the final frame
    cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    save_image("out-" + std::to_string(FRAMES) + ".pam", out_buffer, WIDTH, HEIGHT, 1);

    // Free buffers
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::cout << "Done!" << std::endl;

    return 0;
}