cmake_minimum_required(VERSION 3.20)
project(depth_optimizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros_common_messages REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(Ceres 2.1 REQUIRED)
find_package(TBB REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

#rosidl_generate_interfaces(${PROJECT_NAME}
#  "msg/ObjectData.msg"
#  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
#)


add_executable(depth_optimizer_node
src/depth_optimizer.cpp
src/RobustLinearRegression.cpp
src/DepthOptimizerNode.cpp
src/DepthMapOptimizationProblem.cpp
)

ament_target_dependencies(depth_optimizer_node
 rclcpp
 ros_common_messages
 std_msgs
 sensor_msgs
 geometry_msgs
 cv_bridge
 OpenCV
 Eigen3
 Ceres)

 target_link_libraries(depth_optimizer_node TBB::tbb)
 target_link_libraries(depth_optimizer_node Eigen3::Eigen)
 target_link_libraries(depth_optimizer_node Ceres::ceres)


 target_include_directories(depth_optimizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


set_property(TARGET depth_optimizer_node PROPERTY CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

install(TARGETS
  depth_optimizer_node
  DESTINATION lib/${PROJECT_NAME})



#rosidl_target_interfaces(depth_optimizer_node ${PROJECT_NAME} "rosidl_typesupport_cpp")  #export the interface for the depth_optimizer node, THIS IS VERY IMPORTANT!!!!!

ament_package()
