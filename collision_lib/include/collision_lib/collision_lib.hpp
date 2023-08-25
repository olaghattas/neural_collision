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

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

struct GPUData {

    GPUData(std::vector<float> data, int dim, int stride = 1) {
        size_ = data.size();
        stride_ = stride;
        dim_ = dim;
//        num_elements_in_stride_ = num_elements_in_stride;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
        cudaMemcpy(data_gpu_, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);

        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    GPUData(int size, int dim,  int stride = 1) {
        size_ = size;
        stride_ = stride;
        dim_ = dim;
//        num_elements_in_stride_ = num_elements_in_stride;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
    }

    ~GPUData() {
        cudaFree(data_gpu_);
    }

    int sampleInd(int num_elements) {
//        assert(size_ / dim_ - num_elements >= 0);
        int offset = (std::rand() % (1 + size_ / dim_ - num_elements));
        return offset;
    }

    float *sample(int offset) {
        return (float *) (data_gpu_ + offset * dim_);
    }

    std::vector<float> toCPU(int num_elements_in_stride) {
        num_elements_in_stride_ = num_elements_in_stride;
        std::vector<float> out(size_ * num_elements_in_stride_ / stride_);
        if (stride_ == 1) {
            cudaMemcpy(out.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::vector<float> buf(size_);
            cudaMemcpy(buf.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < size_ / stride_; i++) {
                for (int j = 0; j < num_elements_in_stride_; j++)
                    out[i * num_elements_in_stride_  + j] = buf[stride_ * i + j];
            }
        }

        return out;
    }

    float *data_gpu_;
    int size_;
    int dim_;
    int stride_;
    int num_elements_in_stride_;
};

void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output);

void readMemoryData(std::vector<int>& data, int size);

std::vector<float> check_collision(std::string directoryPath, std::shared_ptr<rclcpp::Node> &node,
                                   std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_);

void check_collision_training(std::string directoryPath, int num_features_);

std::vector<float>
check_collision_inf(std::vector<float> features_inf, std::vector<float> targets_inf, int num_targets, int num_features, std::string path_network_config,
                    std::vector<float> CPU_prams);
//, tcnn::cpp::Module *network_,
//                    std::shared_ptr<rclcpp::Node> &node,
//                    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_);

struct DataPoint {
    float x = 0;
    float y = 0;
    float z = 0;
    float empty = 0;
    float non_empty = 0;

    float door1 = 0;
    float door2 = 0;
    float door3 = 0;
    float door4 = 0;

    DataPoint() {}

    DataPoint(float xIn, float yIn, float zIn, float empty_, float non_empty_, float door1_, float door2_, float door3_, float door4_) {
        x = xIn;
        y = yIn;
        z = zIn;
        float empty = empty_;
        float non_empty = non_empty_;

        float door1 = door1_;
        float door2 = door2_;
        float door3 = door3_;
        float door4 = door4_;
    }

};

typedef std::vector<DataPoint> DataStore;

void write_data(const std::vector<DataPoint> &data);
void publish_pointcloud(const std::vector<float> &features,
                        const std::vector<float> &pred_targets, std::shared_ptr<rclcpp::Node> &node,
                        std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_);
DataStore read_data_from_path(std::string directoryPath );

#endif // GET_COLLISION_HPP
