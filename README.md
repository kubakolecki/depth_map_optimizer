# Depth Map Optimizer
Depth Map Optimizer is a ROS2 package used in NEU-DEPTH project. This package implements sparse map and neural depth map fusion using
least square optimization. In the typical usage scenario, sparse depth comes from visual (or visual-inertial) SLAM.

## Dependencies
This package was tested in Linux with ROS2 Jazzy. Depth Map Optimizer depends on some [messages](https://github.com/kubakolecki/ros_common_messages) that are defined
in a separate ROS2 [package](https://github.com/kubakolecki/ros_common_messages) and need to be built first. Other 3-rd party dependencies: 
- CMake (version >=3.20)
- OpenCV (tested with version 4.12)
- cv_bridge (for converting ROS2 image message to OpenCV cv::Mat)  
- Eigen (tested with version 3.4)
- [Ceres Solver](http://ceres-solver.org/) (tested with version 2.1)  
Ceres Solver has to be built with Suit Sparse to enable Sparse Cholesky decomposition required for reasonably fast optimization.

We used GCC compiler with g++ version 14.3 to support C++23

## Requirements
You need to have rather powerful PC to be able to run the fusion in the real-time (or almost in real-time). In the future we may try to use conjugate gradient approach on GPU that
may provide some time performance boost. We tested this package on laptop with Intel(R) Core(TM) i9-14900HX CPU and 64GB RAM. Typically we achieved fusion runtime about 0.5 s. 
The `'depth_map_scale_factor'` parameter, you can set via launchfiles, impact runtime a lot (the higher the shorter runtime and worse accuracy).

## Building


## Running
