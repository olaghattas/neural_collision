#include <cstdio>
#include <Eigen/Core>

#include <cuda_fp16.h>
//#include "tiny-cuda-nn/cpp_api.h"

#include <collision/get_collision.hpp>

//void readMemoryData(std::vector<int>& data, int size);

void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output) {

    auto batch_size = output.size_ / network->n_output_dims();
    assert(output.size_ == network->n_output_dims() * batch_size);

    output.stride_ = 16;
    tcnn::cpp::Context ctx = network->forward(*stream_ptr, batch_size, inputs.data_gpu_, output.data_gpu_, params,
                                              false);

}


std::vector<float> check_collision( std::vector<float> features_inf, std::vector<float> targets_inf , std::string directoryPath){
    // load data
    auto data = read_data_from_path(directoryPath);

    std::vector<float> features(3 * data.size());
    std::vector<float> targets(2 * data.size());
    for (int i = 0; i < data.size(); i++) {
        auto &datpoint = data[i];
        targets[2 * i + 0] = datpoint.collision;
        targets[2 * i + 1] = datpoint.door_collision;
        features[3 * i + 0] = datpoint.x;
        features[3 * i + 1] = datpoint.y;
        features[3 * i + 2] = datpoint.z;

    }

    GPUData features_gpu{features, 3};
    GPUData targets_gpu{targets, 2};
    GPUData pred_targets_gpu((int) 16 * targets.size(), 1);

    // load config
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto json_file_path = pkg_dir / "config" / "config.json";
    std::fstream file(json_file_path.string());
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);

    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = 1;
    uint32_t batch_size = targets.size() / 2;

    auto stream_ptr = new cudaStream_t();
    cudaStreamCreate(stream_ptr);
    auto trainable_model = tcnn::cpp::create_trainable_model(n_input_dims, n_output_dims, config);


    for (int i = 0; i < 1000; ++i) {
        int ind = features_gpu.sampleInd(batch_size);
        float *training_batch_inputs = features_gpu.sample(ind);
        float *training_batch_targets = targets_gpu.sample(ind);

        auto ctx = trainable_model->training_step(*stream_ptr, batch_size, training_batch_inputs,
                                                  training_batch_targets);
        if (0 == i % 100) {
            float loss = trainable_model->loss(*stream_ptr, ctx);
            std::cout << "iteration=" << i << " loss=" << loss << std::endl;
            auto network_config = config.value("network", nlohmann::json::object());
            auto encoding_config = config.value("encoding", nlohmann::json::object());
            auto network = tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding_config,
                                                                         network_config);
            float *params = trainable_model->params();

        }
    }

    // Inferencing

    auto network = trainable_model->get_network();
    float *params = trainable_model->params();

    GPUData features_gpu_inf{features_inf, 3};
    GPUData pred_targets_inf_gpu((int) 16 * targets_inf.size(), 1);


    predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
    auto pred_targets = pred_targets_gpu.toCPU();

    cudaStreamDestroy(*stream_ptr);

    return pred_targets;
}

std::vector<float> door_collision( std::vector<float> features_inf, std::vector<float> targets_inf , std::string directoryPath){
    // load data
    auto data = read_data_from_path(directoryPath);

    std::vector<float> features(3 * data.size());
    std::vector<float> targets(3 * data.size());
    for (int i = 0; i < data.size(); i++) {
        auto &datpoint = data[i];
        targets[3 * i + 0] = datpoint.door1;
        targets[3 * i + 1] = datpoint.door2;
        targets[3 * i + 2] = datpoint.door3;
        features[3 * i + 0] = datpoint.x;
        features[3 * i + 1] = datpoint.y;
        features[3 * i + 2] = datpoint.z;

    }

    GPUData features_gpu{features, 3};
    GPUData targets_gpu{targets, 3};
    GPUData pred_targets_gpu((int) 16 * targets.size(), 1);

    // load config
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto json_file_path = pkg_dir / "config" / "config.json";
    std::fstream file(json_file_path.string());
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);

    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = 1;
    uint32_t batch_size = targets.size() / 2;

    auto stream_ptr = new cudaStream_t();
    cudaStreamCreate(stream_ptr);
    auto trainable_model = tcnn::cpp::create_trainable_model(n_input_dims, n_output_dims, config);


    for (int i = 0; i < 1000; ++i) {
        int ind = features_gpu.sampleInd(batch_size);
        float *training_batch_inputs = features_gpu.sample(ind);
        float *training_batch_targets = targets_gpu.sample(ind);

        auto ctx = trainable_model->training_step(*stream_ptr, batch_size, training_batch_inputs,
                                                  training_batch_targets);
        if (0 == i % 100) {
            float loss = trainable_model->loss(*stream_ptr, ctx);
            std::cout << "iteration=" << i << " loss=" << loss << std::endl;
            auto network_config = config.value("network", nlohmann::json::object());
            auto encoding_config = config.value("encoding", nlohmann::json::object());
            auto network = tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding_config,
                                                                         network_config);
            float *params = trainable_model->params();

        }
    }

    // Inferencing

    auto network = trainable_model->get_network();
    float *params = trainable_model->params();

    GPUData features_gpu_inf{features_inf, 3};
    GPUData pred_targets_inf_gpu((int) 16 * targets_inf.size(), 1);


    predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
    auto pred_targets = pred_targets_gpu.toCPU();

    cudaStreamDestroy(*stream_ptr);

    return pred_targets;
}
