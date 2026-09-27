# Depth Map Optimizer
Depth Map Optimizer is a ROS2 package used in NEU-DEPTH project. This package implements sparse map and neural depth map fusion using
least square optimization. In the typical usage scenario, sparse depth comes from visual (or visual-inertial) SLAM.

## Dependencies
This package was tested in Linux with ROS2 Jazzy. Depth Map Optimizer depends on some [messages](https://github.com/kubakolecki/ros_common_messages) that are defined
in a separate ROS2 [package](https://github.com/kubakolecki/ros_common_messages) and needs to be built first. Other 3-rd party dependencies: 
- CMake (version >=3.20)
- OpenCV (tested with version 4.12)
- cv_bridge (for converting ROS2 image message to OpenCV cv::Mat)  
- Eigen (tested with version 3.4)
- [Ceres Solver](http://ceres-solver.org/) (tested with version 2.1)  
Ceres Solver has to be built with Suit Sparse to enable Sparse Cholesky decomposition required for reasonably fast optimization.

We used GCC compiler with g++ version 14.3 to support C++23

## Requirements
You need to have rather powerful PC to be able to run the full fusion in the real-time (or almost in real-time). In the future we may try to use conjugate gradient approach on GPU that
may provide some time performance boost. We tested this package on laptop with Intel(R) Core(TM) i9-14900HX CPU and 64GB RAM. Typically we achieved fusion runtime about 0.5 s. 
The `'depth_map_scale_factor'` parameter, you can set via launchfiles, impact runtime a lot (the higher the shorter runtime and worse accuracy). If you set `'do_run_rigorous_optimization'` to `False` then
only the linear depth map correction is applied. It is very fast (like less than 1 ms to few ms) but provides worse accuracy than rigorous optimization. Still this one can be sufficient for your applications
so if you prioritize runtime over accuracy you can always stop computations after linear correction.

## Building
#### About cv_bridge
First you need to have cv_bridge installed. If you have cv_bridge already installed, you can skip this part. cv_bridge is a part of [vision_opencv](https://github.com/ros-perception/vision_opencv) package.
We tested building depth_map_optimizer with cv_bridge installed locally in our ROS2 workspace. At least at the time I write this readme there is no vision_opencv branch dedicated for ROS2 Jazzy distribution so I recommend
building ROS2 Humble branch of cv_bridge (the closest to Jazzy). In your workspace you should have your packages located in the `src` directory, which is a standard way to organize ROS2 workspace.
In the terminal go to the workspace main directory and follow the commands below:
```bash
cd src
git clone https://github.com/ros-perception/vision_opencv.git -b humble
cd ..
colcon build --packages-select cv_bridge
```
#### Building depth_map_optimizer 
Following commands assume your packages are located in the `src` directory
```bash
cd src
git clone git clone https://github.com/kubakolecki/depth_map_optimizer
cd ..
colcon build --packages-select depth_optimizer
source install/setup.bash
```
## Running
depth_map_optimizer subsribes to topic exposing ImageBasedMappingData message. See [messages](https://github.com/kubakolecki/ros_common_messages).
For running the node there is a launchfile prepared: `/ROS2_WORKSPACE/src/depth_map_optimizer/launch/run_depth_optimizer.py`  
Some relevant parameters:  
- `do_run_rigorous_optimization` - If true then the node runs rigorous optimization using Ceres Solver. If flase only linear correction is applied. Applying linear correction only results in a way faster runtime.  
- `depth_map_scale_factor` - Before running rigorous optimization the source depth map will be downsampled by this factor. It does not affect the output depth map. The higher the value the faster the optimization is. Only relevant if `do_run_rigorous_optimization` is true. 
- `regression_outlier_threshold` - Linear correction is computed using RANSAC. This is the outlier threshold to be used in RANSAC. The less accurate the input depth map is, the higher this value should be.  
- `map_point_difference_threshold` - Map points with depth difference from a depth map after linear correction greater than this value won't take part in rigorous optimization.
- `depth_map_uncertainty_coefficient` - We assume the the uncertainty of depth provided by a source depth map is proportional to the depth value. The apriori uncertainty of depth value in pixel location (row, column) in the source depth map will be computed as depth_map_uncertainty_coefficient*depth(row, column)
- `optimization_approach` - if set to WITH_SCALE_ESTIMATION the depth map scale will be explicitly modeled and refined in the optimization process. We avoid this because we found that this does not bring any meaningful improvement but at the same time makes optimization problem  Hessian matrix more dense resulting in prolonged computation time.

For information about other parameters please look in the source code or query them using ROS2 CLI. Ceres loss function parameters [are explained here](http://ceres-solver.org/nnls_modeling.html#lossfunction) in details. The optimization can run in two steps where the loss functions used for sparse map points can be different in each step. We achieve this using Ceres LossFunctionWrapper.
