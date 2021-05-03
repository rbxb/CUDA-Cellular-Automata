#include <iostream>
#include <iomanip>
#include <vector>

#include "mnca.cu"
#include "helpers.cu"

#define FRAMES 1000

int main(void) {
    unsigned char* buf_r;
    unsigned char* buf_w;

    // Allocate buffers
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Create neighborhood 0
    std::vector<int> nh0 = std::vector<int>();
    mnca::generate_nh_midpoint_circle(3, nh0);
    mnca::generate_nh_midpoint_circle(1, nh0);
    int nh0_len = nh0.size() / 2;

    // Copy neighborhood 0 to device
    int* d_nh0;
    cudaMalloc(&d_nh0, nh0_len * sizeof(int) * 2);
    cudaMemcpy(d_nh0, &nh0[0], nh0_len * sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Create neighborhood 1
    std::vector<int> nh1 = std::vector<int>();
    mnca::generate_nh_midpoint_circle(14, nh1);
    mnca::generate_nh_midpoint_circle(11, nh1);
    mnca::generate_nh_midpoint_circle(8, nh1);
    mnca::generate_nh_midpoint_circle(5, nh1);
    int nh1_len = nh1.size() / 2;

    // Copy neighborhood 1 to device
    int* d_nh1;
    cudaMalloc(&d_nh1, nh1_len * sizeof(int) * 2);
    cudaMemcpy(d_nh1, &nh1[0], nh1_len * sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Run seed kernel
    helpers::seed_symmetric << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r);

    // Loop conways game of life
    for (int i = 0; i < FRAMES; i++) {
        // Copy frame from device to host
        cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);

        // Wait for device to finish
        cudaDeviceSynchronize();

        // Start next transition
        mnca::simple_mnca << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (d_nh0, nh0_len, d_nh1, nh1_len, buf_r, buf_w);

        // Update cout
        if (i % 10 == 0) {
            std::cout << i * 100 / FRAMES << "% \t" << i << " of " << FRAMES << std::endl;
        }

        // Save as PPM
        helpers::save_image("out" + helpers::pad_image_index(i) + ".pam", out_buffer, WIDTH, HEIGHT, 1);

        // Swap buffers
        unsigned char* temp = buf_r;
        buf_r = buf_w;
        buf_w = temp;
    }


    // Save the final frame
    cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    helpers::save_image("out" + helpers::pad_image_index(FRAMES) + ".pam", out_buffer, WIDTH, HEIGHT, 1);

    // Free buffers
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::cout << "Done!" << std::endl;

    return 0;
}