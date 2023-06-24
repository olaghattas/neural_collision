#include "ament_index_cpp/get_package_share_directory.hpp"
#include "string"
#include <filesystem>
#include <Eigen/Dense>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

#include <dirent.h>

#pragma once

struct DataPoint {
    float x = 0;
    float y = 0;
    float z = 0;
    bool collision = false;
    bool door_collision = false;

    DataPoint() {}

    DataPoint(float xIn, float yIn, float zIn, bool collisionIn, bool door_collisionIn) {
        x = xIn;
        y = yIn;
        z = zIn;
        collision = collisionIn;
        door_collision = door_collisionIn;
    }
};

typedef std::vector<DataPoint> DataStore;

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


DataStore read_data() {
    DataStore data;

//    std::vector<std::string> filePaths;
    std::string directoryPath = "/home/ola/Desktop/unity_points/";
    DIR* directory;
    struct dirent* entry;

    directory = opendir(directoryPath.c_str());
//    filePaths.push_back("/home/ola/Desktop/collision.json");
//    filePaths.push_back("/home/ola/Desktop/collision_door_points.json");

    if (directory) {
        while ((entry = readdir(directory)) != nullptr) {
            if (entry->d_type == DT_REG) {

                std::ifstream file(directoryPath + entry->d_name);

//                if (!file) {
//                    std::cerr << "Failed to open the file: " << filePath << std::endl;
//                    // Return an empty DataStore or handle the error accordingly
//                    continue;
//                }

                nlohmann::json jsonData;
                file >> jsonData;
                file.close();


                for (const auto &entry: jsonData) {
                    DataPoint item;
                    item.x = entry["posX"];
                    item.y = entry["posY"];
                    item.z = entry["posZ"];
                    item.collision = entry["collision"];
                    if (entry["name_collision"] == "door") {
                        item.door_collision = true;
                    } else {
                        item.door_collision = false;
                    }
                    data.push_back(item);
                }
            }
        }
    }

    std::cout << "Size of data: " << data.size() << std::endl;
    return {data.begin(), data.begin() + 128 * (data.size() / 128)};
}

void printData(const DataStore &data) {
    for (const auto &item: data) {
        std::cout << "x: " << item.x << std::endl;
        std::cout << "y: " << item.y << std::endl;
        std::cout << "z: " << item.z << std::endl;
        std::cout << "collision: " << item.collision << std::endl;
        std::cout << "door_collision: " << (item.door_collision ? "true" : "false") << std::endl;
        std::cout << std::endl;
    }
}