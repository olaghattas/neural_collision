#include <cstdio>
#include <Eigen/Core>

#include <cuda_fp16.h>
//#include "tiny-cuda-nn/cpp_api.h"

#include <collision_lib/collision_lib.hpp>

#include <cuda_runtime.h>  // Add this line to include the CUDA runtime header
#include <iostream>


void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output) {

    auto batch_size = inputs.size_ / network->n_input_dims();

    assert(output.stride_ == 16);
    tcnn::cpp::Context ctx = network->forward(*stream_ptr, batch_size, inputs.data_gpu_, output.data_gpu_, params,
                                              false);

}

DataStore read_data_from_path(std::string directoryPath) {
    DataStore data;

//    std::vector<std::string> filePaths;
//    std::string directoryPath = "/home/ola/Desktop/unity_points/";
    DIR *directory;
    struct dirent *entry;

    directory = opendir(directoryPath.c_str());

    if (directory) {
        while ((entry = readdir(directory)) != nullptr) {
            if (entry->d_type == DT_REG) {
                try {
                    std::ifstream file(directoryPath + entry->d_name);
                    std::string filename = directoryPath + entry->d_name; // Get the filename
                    std::cout << "File name: " << filename << std::endl; // Print file content

                    nlohmann::json jsonData;

                    // debug
                    //file >> jsonData;
                    //std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                    //std::cout << "File Content: " << fileContent << std::endl; // Print file content

                    jsonData = nlohmann::json::parse(file);
                    file.close();

                    for (const auto &entry: jsonData) {
                        DataPoint item;
                        item.x = entry["posX"];
                        item.y = entry["posY"];
                        item.z = entry["posZ"];
                        item.empty = static_cast<float>(entry["empty"]);
                        item.non_empty = static_cast<float>(entry["non_empty"]);
                        item.door1 = static_cast<float>(entry["door1"]);
                        item.door2 = static_cast<float>(entry["door2"]);
                        item.door3 = static_cast<float>(entry["door3"]);
                        item.door4 = static_cast<float>(entry["door4"]);

                        data.push_back(item);
                    }
                } catch (const nlohmann::json::parse_error &e) {
                    std::cerr << "Parsing error: " << e.what() << std::endl;
                }
            }
        }
    }
    std::cout << "Size of data: " << data.size() << std::endl;
    return {data.begin(), data.begin() + 128 * (data.size() / 128)};
}

void write_data(const std::vector<DataPoint> &data) {
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto file_path = pkg_dir / "data";
    std::filesystem::create_directory(file_path);

    std::ofstream file((file_path / "data.bin").string(), std::ios::binary);  // Open the file in binary mode

    int size = data.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(int));
    for (const auto &item: data) {
        file.write(reinterpret_cast<const char *>(&item), sizeof(item));
    }
}

void check_collision_training(std::string directoryPath, int num_features_) {
    /// TODO: have the numb targets not be constant
    // load data
    auto data = read_data_from_path(directoryPath);
    const int num_targets = 6;
    int num_features = num_features_;

    std::vector<float> features(num_features * data.size());
    std::vector<float> targets(num_targets * data.size());
    for (int i = 0; i < data.size(); i++) {
        auto &datpoint = data[i];

        targets[num_targets * i + 0] = datpoint.empty;
        targets[num_targets * i + 1] = datpoint.non_empty;
        targets[num_targets * i + 2] = datpoint.door1;
        targets[num_targets * i + 3] = datpoint.door2;
        targets[num_targets * i + 4] = datpoint.door3;
        targets[num_targets * i + 5] = datpoint.door4;

        features[num_features * i + 0] = datpoint.x;
        features[num_features * i + 1] = datpoint.y;
        features[num_features * i + 2] = datpoint.z;

    }

    GPUData features_gpu{features, num_features};
    GPUData targets_gpu{targets, num_targets};
    GPUData pred_targets_gpu((int) 16 * targets.size() / num_targets, num_targets);

    // load config
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision_lib");
    auto json_file_path = pkg_dir / "config" / "config.json";
    std::cout << "File name: " << json_file_path << std::endl; // Print file content

    std::fstream file(json_file_path.string());
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);

    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = num_targets;
//    uint32_t batch_size = targets.size() / num_targets;
    uint32_t batch_size = 128 * 8 * 8 * 8;

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
        }
    }

    auto network = trainable_model->get_network();
    auto size_ = network->n_params();
    float *params = trainable_model->params();

    std::vector<float> parameters(size_);

    cudaMemcpy(parameters.data(), params, size_ * sizeof(float), cudaMemcpyDeviceToHost);

    nlohmann::json jsonArray = parameters;

    std::ofstream outputFile("/home/olagh/particle_filter/src/neural_collision/collision_lib/config/output.json");
    if (outputFile.is_open()) {
        outputFile << jsonArray.dump(4); // Pretty print with 4 spaces of indentation
        outputFile.close();
        std::cout << "JSON data saved to output.json" << std::endl;
    } else {
        std::cerr << "Unable to open output.json for writing" << std::endl;
    }


    cudaStreamDestroy(*stream_ptr);

    return;
}

std::vector<float> check_collision(std::string directoryPath, std::shared_ptr<rclcpp::Node> &node,
                                   std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_) {
    // load data
    auto data = read_data_from_path(directoryPath);
    const int num_targets = 6;
    int num_features = 3;

    std::vector<float> features(num_features * data.size());
    std::vector<float> targets(num_targets * data.size());
    for (int i = 0; i < data.size(); i++) {
        auto &datpoint = data[i];

        targets[num_targets * i + 0] = datpoint.empty;
        targets[num_targets * i + 1] = datpoint.non_empty;
        targets[num_targets * i + 2] = datpoint.door1;
        targets[num_targets * i + 3] = datpoint.door2;
        targets[num_targets * i + 4] = datpoint.door3;
        targets[num_targets * i + 5] = datpoint.door4;

        features[num_features * i + 0] = datpoint.x;
        features[num_features * i + 1] = datpoint.y;
        features[num_features * i + 2] = datpoint.z;

    }

    GPUData features_gpu{features, num_features};
    GPUData targets_gpu{targets, num_targets};
    GPUData pred_targets_gpu((int) 16 * targets.size() / num_targets, num_targets);

    // load config
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision_lib");
    auto json_file_path = pkg_dir / "config" / "config.json";
    std::cout << "File name: " << json_file_path << std::endl; // Print file content

    std::fstream file(json_file_path.string());
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);

    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = num_targets;
//    uint32_t batch_size = targets.size() / num_targets;
    uint32_t batch_size = 128 * 8 * 8 * 8;

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

    GPUData features_gpu_inf{features, num_features};
    GPUData pred_targets_inf_gpu((int) 16 * targets.size() / num_targets, num_targets, 16);

    for (int i = 0; i < 20; i++) {
        predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
        auto pred_targets = pred_targets_inf_gpu.toCPU(num_targets);
        publish_pointcloud(features, pred_targets, node, pub_);
    }
    cudaStreamDestroy(*stream_ptr);
//    predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
    auto pred_targets = pred_targets_inf_gpu.toCPU(num_targets);


    return pred_targets;
}

std::vector<float>
check_collision_inf(std::vector<float> features_inf, std::vector<float> targets_inf, int num_targets, int num_features, std::string path_network_config,
//                    tcnn::cpp::Module *network_,
                    std::vector<float> CPU_prams
//                   , std::shared_ptr<rclcpp::Node> &node,
//                    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_
) {
    // Inferencing
//    auto network = network_;
    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = 6;
    // load config
//    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision_lib");
//    auto json_file_path = pkg_dir / "config" / "config.json";
    std::cout << "File name: " << path_network_config << std::endl; // Print file content

    std::fstream file(path_network_config);
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);


    auto network_config = config.value("network", nlohmann::json::object());
    auto encoding_config = config.value("encoding", nlohmann::json::object());
    auto network = tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding_config, network_config);
    auto size_ = network->n_params();

    float *params;
    cudaMalloc(&params, size_ * sizeof(float));
    std::vector<float> cpuparams = CPU_prams;
    cudaMemcpy(params, cpuparams.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);

    GPUData features_gpu_inf{features_inf, num_features};
    GPUData pred_targets_inf_gpu((int) 16 * targets_inf.size() / num_targets, num_targets, 16);

    auto stream_ptr = new cudaStream_t();
    cudaStreamCreate(stream_ptr);

//    for (int i = 0; i < 20; i++) {
//        predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
//        auto pred_targets = pred_targets_inf_gpu.toCPU(num_targets);
//        publish_pointcloud(features_inf, pred_targets, node, pub_);
//
//    }
    predict(stream_ptr, network, params, features_gpu_inf, pred_targets_inf_gpu);
    auto pred_targets = pred_targets_inf_gpu.toCPU(num_targets);
//    publish_pointcloud(features_inf, pred_targets, node, pub_);
    cudaStreamDestroy(*stream_ptr);

    return pred_targets;

}

void publish_pointcloud(const std::vector<float> &features,
                        const std::vector<float> &pred_targets, std::shared_ptr<rclcpp::Node> &node,
                        std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_) {

    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    auto x_field = sensor_msgs::msg::PointField();
    auto y_field = sensor_msgs::msg::PointField();
    auto z_field = sensor_msgs::msg::PointField();

    x_field.name = "x";
    x_field.count = 1;
    x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    x_field.offset = 0;

    y_field.name = "y";
    y_field.count = 1;
    y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    y_field.offset = 4;

    z_field.name = "z";
    z_field.count = 1;
    z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    z_field.offset = 8;

    auto color_field = sensor_msgs::msg::PointField();
    color_field.name = "rgba";
    color_field.count = 1;
    color_field.datatype = sensor_msgs::msg::PointField::UINT32;
    color_field.offset = 12;

    msg->width = pred_targets.size();
    msg->height = 1;
    msg->header.stamp = node->get_clock()->now();
    msg->header.frame_id = "map";

    msg->fields = {x_field, y_field, z_field, color_field};
//    msg->is_dense = true;
    msg->point_step = 16;
    msg->row_step = msg->width * msg->point_step;
//    msg->is_bigendian = true;

    msg->data.assign(16 * pred_targets.size(), 0);
    int num_targets = 6;
    for (int i = 0; i < pred_targets.size() / num_targets; i++) {
        int size_ = pred_targets.size();
        int size_f = features.size();

        assert(3 * i + 2 < features.size());
        assert(i * num_targets + 3 < pred_targets.size());
        float point[3] = {features[3 * i + 0], features[3 * i + 1], features[3 * i + 2]};
        memcpy(msg->data.data() + 16 * i, &point[0], 3 * sizeof(float));
        uint8_t color[4] = {0, 0, 0, 255};

        if (pred_targets[i * num_targets] > .5) {
            // empty color blue
            color[3] = 0;
        } else {
            //    color[1] = 255;
            if (pred_targets[i * num_targets + 2] > 0.5) {
                // door 1 color Green
                color[2] = 255;
            } else if (pred_targets[i * num_targets + 3] > 0.5) {
// door 2 color green
                color[1] = 255;
            } else if (pred_targets[i * num_targets + 4] > 0.5) {
// door 3 color purple
                color[2] = 255;
                color[0] = 255;
            } else if (pred_targets[i * num_targets + 5] > 0.5) {
                // door 4 color YELLOW
                color[1] = 255;
                color[0] = 255;
            } else {
// not empty and not doors color light red
                color[0] = 255;
            }
        }
        memcpy(msg->data.data() + 12 + 16 * i, &color[0], 4 * sizeof(uint8_t));
    }
    pub_->publish(*msg);
}