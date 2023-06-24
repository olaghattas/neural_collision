cmake_minimum_required(VERSION 3.8)
project(collision)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)


find_package(fcl REQUIRED)
find_package(CUDA REQUIRED)
find_package(ros2-tiny-cuda-nn REQUIRED)

set(dependencies
        rclcpp
        sensor_msgs
        )
add_executable(main src/main.cpp)
ament_target_dependencies(main ${dependencies})

target_link_libraries(main ${FCL_LIBRARIES})
target_link_libraries(main ${ros2-tiny-cuda-nn_LIBRARIES})
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(main PUBLIC ${ros2-tiny-cuda-nn_INCLUDE_DIR})

target_include_directories(main PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(TARGETS main DESTINATION lib/${PROJECT_NAME})


install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)


ament_package()


# debug
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)

foreach(_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach()