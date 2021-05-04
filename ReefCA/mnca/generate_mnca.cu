#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mnca.cu"
#include "helpers.cu"
#include "save_frame.h"
#include "param_generator.cpp"
#include "nh_generator.cpp"
#include "noise.cu"

void generate_mnca(ParamGenerator * paramg, int sims, int frames, SaveFrame * savef, std::string name = "") {

    // Seed rand
    auto now = std::chrono::system_clock::now();
    srand(std::chrono::system_clock::to_time_t(now));

    unsigned char* buf_r;
    unsigned char* buf_w;

    // Allocate buffers
    cudaMalloc(&buf_r, SIZE);
    cudaMalloc(&buf_w, SIZE);

    // Create out buffer
    unsigned char* out_buffer = new unsigned char[SIZE];

    // Create an array to store params
    unsigned int params[NUM_PARAMS];

    // Create a vector to store parameters from each sim
    unsigned int* all_params = new unsigned int[sims * NUM_PARAMS];

    for (int sim = 0; sim < sims; sim++) {

        std::cout << "Sim " << sim + 1 << " of " << sims << std::endl;

        // Create neighborhood 0
        std::vector<int> nh0 = generate_nh(5, 4);
        int nh0_len = nh0.size() / 2;

        // Copy neighborhood 0 to device
        int* d_nh0;
        cudaMalloc(&d_nh0, nh0_len * sizeof(int) * 2);
        cudaMemcpy(d_nh0, &nh0[0], nh0_len * sizeof(int) * 2, cudaMemcpyHostToDevice);

        // Create neighborhood 1
        std::vector<int> nh1 = generate_nh(15, 10);
        int nh1_len = nh1.size() / 2;

        // Copy neighborhood 1 to device
        int* d_nh1;
        cudaMalloc(&d_nh1, nh1_len * sizeof(int) * 2);
        cudaMemcpy(d_nh1, &nh1[0], nh1_len * sizeof(int) * 2, cudaMemcpyHostToDevice);

        // MNCA parameters
        paramg->generate_params(nh0_len, NUM_PARAMS_NH0, &params[0]);
        paramg->generate_params(nh1_len, NUM_PARAMS_NH1, &params[NUM_PARAMS_NH0]);
        cudaMemcpyToSymbol(mnca::d_params, params, NUM_PARAMS * sizeof(unsigned int));

        for (int p = 0; p < NUM_PARAMS; p++) {
            all_params[sim * NUM_PARAMS + p] = params[p];
        }

        // Run seed kernel
        wave_noise << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (buf_r);

        // Loop conways game of life
        for (int i = 0; i <= frames; i++) {
            // Start next transition
            mnca::simple_mnca << < (SIZE + THREADS - 1) / THREADS, THREADS >> > (d_nh0, nh0_len, d_nh1, nh1_len, buf_r, buf_w);

            // Swap buffers
            unsigned char* temp = buf_r;
            buf_r = buf_w;
            buf_w = temp;
        }

        // Save the final frame
        cudaMemcpy(out_buffer, buf_r, SIZE, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        helpers::save_image(name + helpers::pad_image_index(sim) + ".pam", out_buffer, WIDTH, HEIGHT, 1);
    }

    // Free buffers
    cudaFree(buf_r);
    cudaFree(buf_w);

    std::ofstream ofs;
    ofs.open(name + ".csv", std::ios::binary);
    for (int i = 0; i < sims; i++) {
        ofs << i << "    ";
        for (int p = 0; p < NUM_PARAMS; p++) {
            ofs << all_params[i * NUM_PARAMS + p] << ","; 
        }
        ofs << std::endl;
    }
    ofs.close();

    std::cout << "Done!" << std::endl;
}
