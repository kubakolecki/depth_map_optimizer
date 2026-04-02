#pragma once

#include <rclcpp/rclcpp.hpp>
#include "depth_optimizer/msg/object_data.hpp"
#include "depth_optimizer/RobustLinearRegression.hpp"
#include "depth_optimizer/DepthMapOptimizationConfig.hpp"

#include <opencv2/opencv.hpp>

#include <memory>

#include <filesystem>


class DepthOptimizerNode: public rclcpp::Node
{

private:
    rclcpp::Subscription<depth_optimizer::msg::ObjectData>::SharedPtr objectDataSubscriber;
    void objectDataCallback(const depth_optimizer::msg::ObjectData::SharedPtr msg);
    depth_map_optimization::LossFunctionDescription createLossFunctionDescription(const std::string& lossFunctionString, const double parameter) const;
    std::unique_ptr<RobustLinearRegression> m_robustLinearRegression;

    cv::Mat m_imageCoordinatesY; 
    cv::Mat m_imageCoordinatesX;
    cv::Mat m_ones;
    
    bool m_isTheFirstFrameBeingProcessed{true};
    int m_frameCounter{0};
    depth_map_optimization::DepthMapOptimizationConfig m_depthMapOptimizationConfig;

    std::filesystem::path m_path_depth_maps_original{"/home/kuba/dev/projects/ros2_jazzy_vimbax_ws/depth_maps_original"};
    std::filesystem::path m_path_depth_maps_after_linear_correction{"/home/kuba/dev/projects/ros2_jazzy_vimbax_ws/depth_maps_after_linear_correction"};
    std::filesystem::path m_path_depth_maps_optimized{"/home/kuba/dev/projects/ros2_jazzy_vimbax_ws/depth_maps_optimized"};

public:

    DepthOptimizerNode();


};
