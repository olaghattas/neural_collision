cmake_minimum_required(VERSION 3.8)
project(collision_lib)
enable_language(CUDA)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(ros2-tiny-cuda-nn REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(dependencies
        ament_index_cpp
        )

add_library(collision_lib src/collision_lib.cpp)
target_link_libraries(collision_lib
        ${ros2-tiny-cuda-nn_LIBRARIES}
        )
ament_target_dependencies(collision_lib ${dependencies})
target_include_directories(collision_lib PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${ros2-tiny-cuda-nn_INCLUDE_DIR})

target_include_directories(collision_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


install(DIRECTORY include/
        DESTINATION include/
        )

install(
        TARGETS collision_lib
        EXPORT collision_libTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(
        DIRECTORY config
        DESTINATION share/${PROJECT_NAME}

)

ament_export_targets(collision_libTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
        ament_index_cpp
)
ament_export_include_directories(include)
ament_export_libraries(${ros2-tiny-cuda-nn_LIBRARIES})


ament_package()
