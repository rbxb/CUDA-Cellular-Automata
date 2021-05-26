/*
 * conway.cu
 *
 * https://github.com/rbxb/ReefCA
 */

#include "conway.cuh"

#include "cudahelpers.cuh"

using namespace ReefCA;

template<int width, int height, int depth, typename T>
__global__ void ReefCA::conway_transition(T* buf_r, T* buf_w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int x = i % width;
        int y = i / height;
        unsigned char count = 0;
        count += buf_r[get_rel<width, height, depth>(x, y, 1, 1)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, 1, 0)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, 1, -1)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, 0, 1)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, 0, -1)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, -1, 1)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, -1, 0)] & 1;
        count += buf_r[get_rel<width, height, depth>(x, y, -1, -1)] & 1;
        if (count == 3) buf_w[i * depth] = 255;
        else if (count != 2) buf_w[i * depth] = 0;
        else buf_w[i * depth] = buf_r[i * depth];
    }
}