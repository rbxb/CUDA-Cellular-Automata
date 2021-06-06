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
__device__ unsigned long int ReefCA::sum_nhood(T* buf, int x, int y, nhood* nh, T threshold) {
    unsigned long int sum = 0;
    int* p = nh->p;
    for (int i = 0; i < nh->size; i++) {
        int nx = p[i * 2];
        int ny = p[i * 2 + 1];
        int pix = get_rel<width, height, depth>(x, y, nx, ny);
        T value = buf[pix];
        if (value > threshold) {
            sum += value;
        }
    }
    return sum;
}

template<int width, int height, int depth, typename T, int max_nhs>
__global__ void ReefCA::mnca_transition(T* buf_r, T* buf_w, nhood* nhs, rule<T>* rules, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int x = i % width;
        int y = i / width;

        unsigned long int sums [max_nhs];
        unsigned char summed = 0;

        for (int k = 0; k < n; k++) {
            rule<T> r = rules[k];
            
            // check if the neighborhood has been summed already
            // if not, sum it and cache the result
            if ((summed & (1 << r.nh)) == 0) {
                sums[r.nh] = sum_nhood<width, height, depth, T>(buf_r, x, y, &nhs[r.nh]);
                summed |= 1 << r.nh;
            }

            // apply rule
            if (r.lower <= sums[r.nh] && sums[r.nh] <= r.upper) {
                buf_w[i] = r.value;
                return;
            }
        }

        buf_w[i * depth] = buf_r[i];
    }
}

template<int width, int height, int depth, typename T>
__global__ void ReefCA::draw_nhood(T* buf, int x, int y, nhood* nh) {
    for (int i = 0; i < nh->size; i++) {
        int nx = nh->p[i * 2];
        int ny = nh->p[i * 2 + 1];
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

template<typename T>
bool ReefCA::read_mnca_rule(nhood** nhs, int* num_nhs, rule<T>** rules, int* num_rules) {
    std::vector<std::vector<int>> neighborhoods_vector = std::vector<std::vector<int>>();
    std::vector<ReefCA::rule<unsigned char>> rules_vector = std::vector<ReefCA::rule<unsigned char>>();

    // Read in from input
    bool reading_params = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "params") {
            reading_params = true;
            continue;
        } else if (reading_params) {
            std::stringstream ss(line);
            int n, lower, upper, value;
            ss >> n >> lower >> upper >> value;
            ReefCA::rule<unsigned char> r = { 
                n,
                unsigned long int(lower * T(-1)), 
                unsigned long int(upper * T(-1)), 
                T(value)
            };
            rules_vector.push_back(r);
        } else {
            std::stringstream ss(line);
            std::vector<int> nh = std::vector<int>();
            int v;
            while (ss >> v) {
                nh.push_back(v);
            }
            neighborhoods_vector.push_back(nh);
        }
    }

    // Upload neighborhoods
    std::vector<nhood> nhoods = std::vector<nhood>();
    for (int i = 0; i < neighborhoods_vector.size(); i++) {
        nhoods.push_back(ReefCA::upload_nh(neighborhoods_vector[i]));
    }
    cudaMalloc(nhs, sizeof(nhood) * nhoods.size());
    cudaMemcpy(*nhs, &nhoods[0], sizeof(nhood) * nhoods.size(), cudaMemcpyHostToDevice);
    *num_nhs = nhoods.size();

    // Upload MNCA rules
    cudaMalloc(rules, rules_vector.size() * sizeof(rule<unsigned char>));
    cudaMemcpy(*rules, &rules_vector[0], rules_vector.size() * sizeof(rule<unsigned char>), cudaMemcpyHostToDevice);
    *num_rules = rules_vector.size();

    return true;
}

void ReefCA::free_nhs_values(nhood* nhs, int n) {
    nhood* host_nhs = new nhood [n];
    cudaMemcpy(host_nhs, nhs, sizeof(nhood) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        cudaFree(host_nhs[i].p);
    }
    delete host_nhs;
}