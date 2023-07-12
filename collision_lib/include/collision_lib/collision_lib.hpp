//
// Created by ola on 7/9/23.
//

#ifndef GET_COLLISION_HPP
#define GET_COLLISION_HPP

#include <vector>
#include <string>
#include "tiny-cuda-nn/cpp_api.h"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "string"
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <dirent.h>

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

struct DataPoint {
    float x = 0;
    float y = 0;
    float z = 0;
    bool collision = false;
    bool door_collision = false;

    // number of door depending on the number if doors in the house.
    bool door1 = false;
    bool door2 = false;
    bool door3 = false;

    DataPoint() {}

    DataPoint(float xIn, float yIn, float zIn, bool collisionIn, bool door_collisionIn) {
        x = xIn;
        y = yIn;
        z = zIn;
        collision = collisionIn;
        door_collision = door_collisionIn;
    }
    // switched order of xyz and doors so we dont have the same constructor in houses with only two doors
    DataPoint(bool door1In, bool door2In, bool door3In, float xIn, float yIn, float zIn) {
        x = xIn;
        y = yIn;
        z = zIn;
        door1 = door1In;
        door2 = door2In;
        door3 = door3In;
    }
};

typedef std::vector<DataPoint> DataStore;

void write_data(const std::vector<DataPoint> &data);

DataStore read_data_from_path(std::string directoryPath );

#endif // GET_COLLISION_HPP
