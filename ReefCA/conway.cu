#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "contants.h"
#include "helpers.cu"

// Kernel function for game of life transition
__global__
void transition(char* buf_r, char* buf_w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE) {
        int x = i % WIDTH;
        int y = i / HEIGHT;
        char count = 0;
        count += buf_r[get_rel(x, y, 1, 1)] & 1;
        count += buf_r[get_rel(x, y, 1, 0)] & 1;
        count += buf_r[get_rel(x, y, 1, -1)] & 1;
        count += buf_r[get_rel(x, y, 0, 1)] & 1;
        count += buf_r[get_rel(x, y, 0, -1)] & 1;
        count += buf_r[get_rel(x, y, -1, 1)] & 1;
        count += buf_r[get_rel(x, y, -1, 0)] & 1;
        count += buf_r[get_rel(x, y, -1, -1)] & 1;
        if (count == 3) buf_w[i] = 255;
        else if (count != 2) buf_w[i] = 0;
        else buf_w[i] = buf_r[i];
    }
}