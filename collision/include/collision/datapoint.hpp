#include "ament_index_cpp/get_package_share_directory.hpp"
#include "string"
#include <filesystem>
#include <eigen3/Eigen/Dense>
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

DataStore read_data_from_path(std::string directoryPath ) {
    DataStore data;

//    std::vector<std::string> filePaths;
//    std::string directoryPath = "/home/ola/Desktop/unity_points/";
    DIR* directory;
    struct dirent* entry;

    directory = opendir(directoryPath.c_str());

    if (directory) {
        while ((entry = readdir(directory)) != nullptr) {
            if (entry->d_type == DT_REG) {

                std::ifstream file(directoryPath + entry->d_name);

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

DataStore read_data() {
    DataStore data;
//    std::vector<std::string> filePaths;
    std::string directoryPath = "/home/ola/Desktop/unity_points/";
    DIR* directory;
    struct dirent* entry;

    directory = opendir(directoryPath.c_str());

    if (directory) {
        while ((entry = readdir(directory)) != nullptr) {
            if (entry->d_type == DT_REG) {

                std::ifstream file(directoryPath + entry->d_name);

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