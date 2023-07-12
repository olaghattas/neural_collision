//
// Created by ola on 7/9/23.
//


#ifndef GET_COLLISION_HPP
#define GET_COLLISION_HPP

#include <vector>
#include <string>
#include "tiny-cuda-nn/cpp_api.h"
#include <collision/datapoint.hpp>

struct GPUData {
    GPUData(std::vector<float> data, int dim) {
        size_ = data.size();
        stride_ = 1;
        dim_ = dim;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
        cudaMemcpy(data_gpu_, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);

        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    GPUData(int size, int dim) {
        size_ = size;
        stride_ = 1;
        dim_ = dim;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
    }

    ~GPUData() {
        cudaFree(data_gpu_);
    }

    int sampleInd(int num_elements) {
        assert(size_ / dim_ - num_elements >= 0);
        int offset = (std::rand() % (1 + size_ / dim_ - num_elements));
        return offset;
    }

    float *sample(int offset) {
        return (float *) (data_gpu_ + offset * dim_);
    }

    std::vector<float> toCPU() {
        std::vector<float> out(size_ / stride_);
        if (stride_ == 1) {
            cudaMemcpy(out.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::vector<float> buf(size_);
            cudaMemcpy(buf.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < size_ / stride_; i++) {
                out[i] = buf[stride_ * i];
            }
        }

        return out;
    }

    float *data_gpu_;
    int size_;
    int dim_;
    int stride_;
};

void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output);

void readMemoryData(std::vector<int>& data, int size);

std::vector<float> check_collision(std::vector<float> features_inf, std::vector<float> targets_inf, std::string directoryPath);

std::vector<float> door_collision(std::vector<float> features_inf, std::vector<float> targets_inf, std::string directoryPath);


#endif // GET_COLLISION_HPP
