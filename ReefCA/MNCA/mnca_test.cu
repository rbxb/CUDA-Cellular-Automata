#include <iostream>
#include <iomanip>

#include "reefca.h"

#define FRAMES SIZE

int main(void) {
    unsigned char* buf_w;

    // Allocate buffers
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Create neighborhood
    std::vector<int> v = std::vector<int>();
    ReefCA::generate_nh_fill_circle(7, 3, v);
    nhood* nhoods = ReefCA::create_nhood_row(v);

    for (int i = 0; i < 1000; i++) {
        int x = i % WIDTH;
        int y = i / WIDTH;

        // Draw neighborhood
        ReefCA::draw_nhood << < 1, 1 >> > (buf_w, y, nhoods[x]);

        // Copy frame from device to host
        cudaMemcpy(out_buffer, buf_w, SIZE, cudaMemcpyDeviceToHost);

        // Wait for device to finish
        cudaDeviceSynchronize();

        // Save as PPM
        ReefCA::save_pam("out" + ReefCA::pad_image_index(i) + ".pam", out_buffer);
    }
    
    // Free buffers
    cudaFree(buf_w);
    for (int i = 0; i < WIDTH; i++) {
        nhood nh = nhoods[i];
        if (nh.offset == 0) cudaFree(nh.p);
    }

    return 0;
}