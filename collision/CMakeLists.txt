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
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fcl REQUIRED)
find_package(CUDA REQUIRED)
find_package(ros2-tiny-cuda-nn REQUIRED)

set(dependencies
        rclcpp
        sensor_msgs
        ament_index_cpp
        )
add_executable(main src/main.cpp)
ament_target_dependencies(main ${dependencies})

target_link_libraries(main ${FCL_LIBRARIES})
target_link_libraries(main ${ros2-tiny-cuda-nn_LIBRARIES})
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_include_directories(main PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(main PUBLIC ${ros2-tiny-cuda-nn_INCLUDE_DIR})

target_include_directories(main PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(TARGETS main DESTINATION lib/${PROJECT_NAME})

add_library(get_collision SHARED ../collision_lib/src/collision_lib.cpp)
ament_target_dependencies(get_collision ${dependencies})

#target_link_libraries(get_collision ${FCL_LIBRARIES})
target_link_libraries(get_collision ${ros2-tiny-cuda-nn_LIBRARIES})
target_include_directories(get_collision PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_include_directories(get_collision PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(get_collision PUBLIC ${ros2-tiny-cuda-nn_INCLUDE_DIR})

target_include_directories(get_collision PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

install(TARGETS get_collision DESTINATION lib/${PROJECT_NAME})

install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}

)

install(DIRECTORY include/
        DESTINATION include/
        )

#install(FILES src/get_collision.cpp
#        DESTINATION include/collision/
#        )

# use if this is only a library
#install(
#        TARGETS pddl_parser
#        EXPORT pddl_parserTargets
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin)

#ament_export_targets(pddl_parserTargets HAS_LIBRARY_TARGET)

#ament_export_dependencies(
#)
ament_export_include_directories(include)

ament_package()


# debug
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)

foreach(_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach()