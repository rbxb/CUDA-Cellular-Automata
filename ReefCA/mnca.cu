/*
 * mnca.cu
 *
 * https://github.com/rbxb/ReefCA
 */

#include "mnca.cuh"

#include <algorithm>
#include <iostream>
#include "cudahelpers.cuh"

using namespace ReefCA;

template<int width, int height, int depth, typename T>
__device__ unsigned long int ReefCA::sum_nhood(T* buf, int x, int y, nhood nh, T threshold) {
    unsigned long int sum = 0;
    for (int i = 0; i < nh.size; i++) {
        int nx = nh.p[i * 2];
        int ny = nh.p[i * 2 + 1];
        int pix = get_rel<width, height, depth>(x, y, nx, ny);
        T value = buf[pix];
        if (value > threshold) {
            sum += value;
        }
    }
    return sum;
}

template<int width, int height, int depth, typename T>
__global__ void ReefCA::mnca_2n_8t(T* buf_r, T* buf_w, nhood nh0, nhood nh1, unsigned short int* params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int x = i % width;
        int y = i / height;
        T max = T(-1);

        unsigned long int sum0 = sum_nhood<width, height, depth, T>(buf_r, x, y, nh0) / unsigned long int(max);
        if (sum0 < params[0]) {
            buf_w[i] = 0;
            return;
        } else if (params[1] <= sum0 && sum0 < params[2]) {
            buf_w[i] = max;
            return;
        } else if (params[3] <= sum0) {
            buf_w[i] = 0;
            return;
        }
        
        unsigned long int sum1 = sum_nhood<width, height, depth, T>(buf_r, x, y, nh1) / unsigned long int(max);
        if (sum1 < params[4]) {
            buf_w[i] = 0;
            return;
        } else if (params[5] <= sum1 && sum1 < params[6]) {
            buf_w[i] = max;
            return;
        } else if (params[7] <= sum1) {
            buf_w[i] = 0;
            return;
        }

        buf_w[i] = buf_r[i];
    }
}

template<int width, int height, int depth, typename T>
__global__ void ReefCA::draw_nhood(T* buf, int x, int y, nhood nh) {
    for (int i = 0; i < nh.size; i++) {
        int nx = nh.p[i * 2];
        int ny = nh.p[i * 2 + 1];
        int pix = get_rel<width, height, depth>(x, y, nx, ny);
        buf[pix] = T(-1);
    }
}

nhood ReefCA::upload_nh(std::vector<int>& v) {
    int* p;
    cudaMalloc(&p, v.size() * sizeof(int));
    cudaMemcpy(p, &v[0], v.size() * sizeof(int), cudaMemcpyHostToDevice);
    return nhood{p, int(v.size()) / 2};
}

void ReefCA::generate_nh_fill_circle(int r_outer, int r_inner, std::vector<int>& v) {
    for (int x = -r_outer; x <= r_outer; x++) {
        for (int y = -r_outer; y <= r_outer; y++) {
            double d = dist(x, y);
            if (d <= r_outer && d > r_inner) {
                v.push_back(x);
                v.push_back(y);
            }
        }
    }
}