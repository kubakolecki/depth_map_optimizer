#pragma once

#include <rclcpp/rclcpp.hpp>
#include "depth_optimizer/msg/object_data.hpp"
#include "depth_optimizer/RobustLinearRegression.h"

#include <opencv2/opencv.hpp>

#include <memory>

class DepthOptimizerNode: public rclcpp::Node
{

private:
    rclcpp::Subscription<depth_optimizer::msg::ObjectData>::SharedPtr objectDataSubscriber;
    void objectDataCallback(const depth_optimizer::msg::ObjectData::SharedPtr msg);

    std::unique_ptr<RobustLinearRegression> m_robustLinearRegression;

    cv::Mat m_imageCoordinatesY; 
    cv::Mat m_imageCoordinatesX;
    cv::Mat m_ones;
    
    bool m_isTheFirstFrameBeingProcessed{true};

    int m_frameCounter{0};

public:

    DepthOptimizerNode();


};
