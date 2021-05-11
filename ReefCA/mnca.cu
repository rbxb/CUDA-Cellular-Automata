#ifndef MNCA_CU
#define MNCA_CU

#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helpers.cu"
#include "contants.h"

#define NUM_PARAMS_NH0 4
#define NUM_PARAMS_NH1 4
const int NUM_PARAMS = NUM_PARAMS_NH0 + NUM_PARAMS_NH1;

namespace mnca {

    int compare_nh_points(const void* a, const void* b) {
        int a_x = ((int*)a)[0];
        int a_y = ((int*)a)[1];
        int b_x = ((int*)b)[0];
        int b_y = ((int*)b)[1];
        int a_i = a_y * WIDTH + a_x;
        int b_i = b_y * WIDTH + b_x;
        return b_i - a_i;
    }

    // Sorts a neighborhood by the pixel index
    void sort_nh(std::vector<int> a) {
        std::qsort(&a[0], a.size() / 2, sizeof(int) * 2, compare_nh_points);
    }

    // Generates a list of x and y values for a disk with inner and outer radius
    void generate_nh_fill_circle(int r_outer, int r_inner, std::vector<int> &v) {
        for (int x = -r_outer; x <= r_outer; x++) {
            for (int y = -r_outer; y <= r_outer; y++) {
                double d = helpers::dist(x, y);
                if (d <= r_outer && d > r_inner) {
                    v.push_back(x);
                    v.push_back(y);
                }
            }
        }
    }

    // Generates a list of x and y values for a square with inner and outer radius
    void generate_nh_fill_square(int r_outer, int r_inner, std::vector<int> &v) {
        for (int x = -r_outer; x <= r_outer; x++) {
            for (int y = -r_outer; y <= r_outer; y++) {
                if (abs(x) > r_inner || abs(y) > r_inner) {
                    v.push_back(x);
                    v.push_back(y);
                }
            }
        }
    }

    // Generates a neighborhood using midpoint circle algorithm
    void generate_nh_midpoint_circle(int r, std::vector<int>& v) {
        int x = r, y = 0, p = 1 - r;

        v.push_back(r);
        v.push_back(0);

        v.push_back(-r);
        v.push_back(0);

        v.push_back(0);
        v.push_back(r);

        v.push_back(0);
        v.push_back(-r);

        while (x > y) {
            y++;

            if (p <= 0) {
                p = p + 2 * y + 1;
            } else {
                x--;
                p = p + 2 * y - 2 * x + 1;
            }
            
            if (x < y) {
                break;
            }

            v.push_back(x);
            v.push_back(y);

            v.push_back(-x);
            v.push_back(y);

            v.push_back(x);
            v.push_back(-y);

            v.push_back(-x);
            v.push_back(-y);

            if (x != y) {
                v.push_back(y);
                v.push_back(x);

                v.push_back(-y);
                v.push_back(x);

                v.push_back(y);
                v.push_back(-x);

                v.push_back(-y);
                v.push_back(-x);
            }
        }
    }

    // Sums a neighborhood
    // x,y is the center point
    // nh contains the x,y offsets of each neighbor
    // len is the number of x,y pairs in nh
    // buf is the buffer to read from
    // threshold is the minimum threshold for a neighbor to be added to the sum
    template<typename T = unsigned char>
    __device__
    unsigned long int sum_nh(int x, int y, int* nh, int len, T* buf, T threshold = 0) {
        unsigned long int sum = 0;
        for (int i = 0; i < len; i++) {
            int nhx = nh[i * 2];
            int nhy = nh[i * 2 + 1];
            int pixel_index = helpers::get_rel(x, y, nhx, nhy);
            T value = buf[pixel_index];
            if (value > threshold) {
                sum += value;
            }
        }
        return sum;
    }

#define ROW_LENGTH 4

    // Sums a neighborhood
    // x,y is the center point
    // nh contains the x,y offsets of each neighbor
    // len is the number of x,y pairs in nh
    // buf is the buffer to read from
    // threshold is the minimum threshold for a neighbor to be added to the sum
    template<typename T = unsigned char>
    __device__
    void sum_nh_row(int x, int y, int* nh, int len, T* buf, unsigned long int* sums, T threshold = 0) {
        for (int i = 0; i < len; i++) {
            int nhx = nh[i * 2];
            int nhy = nh[i * 2 + 1];
            int pixel_index = helpers::get_rel(x, y, nhx, nhy);
            for (int x_shift = 0; x_shift < ROW_LENGTH; x_shift++) {
                T value = buf[pixel_index + x_shift];
                if (value > threshold) {
                    sums[x_shift] += 1;
                }
            }
        }
    }

    // A simple MNCA transition
    __constant__ unsigned int d_params[NUM_PARAMS];
    __global__
    void simple_mnca(int* nh0, int nh0_len, int* nh1, int nh1_len, unsigned char* buf_r, unsigned char* buf_w) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < SIZE) {
            int x = i % WIDTH;
            int y = i / WIDTH;

            // Sum nh0
            unsigned long int sum_nh0 = sum_nh(x, y, nh0, nh0_len, buf_r) / 255;

            if (sum_nh0 < d_params[0]) {
                buf_w[i] = 0;
                return;
            } else if (d_params[1] <= sum_nh0 && sum_nh0 < d_params[2]) {
                buf_w[i] = 255;
                return;
            } else if (d_params[3] <= sum_nh0) {
                buf_w[i] = 0;
                return;
            }

            // Sum nh1
            unsigned long int sum_nh1 = sum_nh(x, y, nh1, nh1_len, buf_r) / 255;
            if (sum_nh1 < d_params[4]) {
                buf_w[i] = 0;
                return;
            }
            else if (d_params[5] <= sum_nh1 && sum_nh1 < d_params[6]) {
                buf_w[i] = 255;
                return;
            }
            else if (d_params[7] <= sum_nh1) {
                buf_w[i] = 0;
                return;
            }

            buf_w[i] = buf_r[i];
        }
    }

    // A simple MNCA transition
    __global__
        void simple_mnca_row(int* nh0, int nh0_len, int* nh1, int nh1_len, unsigned char* buf_r, unsigned char* buf_w) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * ROW_LENGTH;
        if (i < SIZE) {
            int x = i % WIDTH;
            int y = i / WIDTH;     

            // Sum nh1
            unsigned long int sum[ROW_LENGTH];
            sum_nh_row(x, y, nh1, nh1_len, buf_r, &sum[0]);

            for (int x_shift = 0; x_shift < ROW_LENGTH; x_shift++) {
                if (sum[x_shift] < d_params[4]) {
                    buf_w[i + x_shift] = 0;
                }
                else if (d_params[5] <= sum[x_shift] && sum[x_shift] < d_params[6]) {
                    buf_w[i + x_shift] = 255;
                }
                else if (d_params[7] <= sum[x_shift]) {
                    buf_w[i + x_shift] = 0;
                }
            }

            // Sum nh0
            sum_nh_row(x, y, nh0, nh0_len, buf_r, &sum[0]);

            for (int x_shift = 0; x_shift < ROW_LENGTH; x_shift++) {
                if (sum[x_shift] < d_params[0]) {
                    buf_w[i + x_shift] = 0;
                }
                else if (d_params[1] <= sum[x_shift] && sum[x_shift] < d_params[2]) {
                    buf_w[i + x_shift] = 255;
                }
                else if (d_params[3] <= sum[x_shift]) {
                    buf_w[i + x_shift] = 0;
                }
                buf_w[i + x_shift] = buf_r[i + x_shift];
            }
        }
    }
};

#endif // MNCA_CU