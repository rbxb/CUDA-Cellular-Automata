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
    nhood nh = ReefCA::upload_nh(v);

    // Draw neighborhood
    ReefCA::draw_nhood << < 1, 1 >> > (buf_w, 0, 0, nh);

    // Copy frame from device to host
    cudaMemcpy(out_buffer, buf_w, SIZE, cudaMemcpyDeviceToHost);

    // Wait for device to finish
    cudaDeviceSynchronize();

    // Save as PPM
    ReefCA::save_pam("mnca_test.pam", out_buffer);
    
    // Free buffers
    cudaFree(buf_w);
    cudaFree(nh.p);

    return 0;
}