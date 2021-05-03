#include <iostream>
#include <iomanip>

#include "mnca.cu"
#include "helpers.cu"

#define FRAMES 100

__global__
void draw_nh(int x, int y, int* nh, int len, unsigned char* buf_w) {
    for (int i = 0; i < len; i++) {
        int nhx = nh[i * 2];
        int nhy = nh[i * 2 + 1];
        int index = helpers::get_rel(x,y,nhx,nhy);
        buf_w[index] = 255;
    }
}

int main(void) {
    unsigned char* buf_w;

    // Allocate buffers
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Create neighborhood
    std::vector<int> v = std::vector<int>();
    mnca::generate_nh_midpoint_circle(14, v);
    mnca::generate_nh_midpoint_circle(11, v);
    mnca::generate_nh_midpoint_circle(8, v);
    mnca::generate_nh_midpoint_circle(5, v);
    int len = v.size() / 2;
    
    // Copy neighborhood to device
    int* d_nh;
    cudaMalloc(&d_nh, len * sizeof(int) * 2);
    cudaMemcpy(d_nh, &v[0], len * sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Draw neighborhood
    draw_nh<<< 1, 1 >>>(40, 2, d_nh, len, buf_w);

    // Copy frame from device to host
    cudaMemcpy(out_buffer, buf_w, SIZE, cudaMemcpyDeviceToHost);

    // Wait for device to finish
    cudaDeviceSynchronize();

    // Save as PPM
    helpers::save_image("mnca_test.pam", out_buffer, WIDTH, HEIGHT, 1);

    // Free buffers
    cudaFree(buf_w);

    return 0;
}