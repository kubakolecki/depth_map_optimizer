#pragma once

#include <rclcpp/rclcpp.hpp>
#include "ros_common_messages/msg/image_based_mapping_data.hpp"
#include "ros_common_messages/msg/object_localization_info.hpp"
#include "depth_optimizer/RobustLinearRegression.hpp"
#include "depth_optimizer/DepthMapOptimizationConfig.hpp"

#include <opencv2/opencv.hpp>

#include <memory>

#include <filesystem>

struct ObjectDepthStatistics
{
    double meanDepth{0.0};
    double medianDepth{0.0};
    double minDepth{0.0};
    double maxDepth{0.0};
};


class DepthOptimizerNode: public rclcpp::Node
{

private:
    rclcpp::Subscription<ros_common_messages::msg::ImageBasedMappingData>::SharedPtr imageBasedMappingDataSubscriber;
    rclcpp::Publisher<ros_common_messages::msg::ObjectLocalizationInfo>::SharedPtr objectLocalizationInfoPublisher;
    void imageBasedMappingDataCallback(const ros_common_messages::msg::ImageBasedMappingData::SharedPtr msg);
    depth_map_optimization::LossFunctionDescription createLossFunctionDescription(const std::string& lossFunctionString, const double parameter) const;
    cv::Point2f computeLabelCentroid(const cv::Mat& labels, uint16_t label) const;
    ObjectDepthStatistics computeObjectDepthStatistics(const cv::Mat& depthMap, const cv::Mat& labels, uint16_t label) const;
    
    std::unique_ptr<RobustLinearRegression> m_robustLinearRegression;

    cv::Mat m_imageCoordinatesY; 
    cv::Mat m_imageCoordinatesX;
    cv::Mat m_ones;

    int m_minNumberOfMapPoints{20};
    
    bool m_isTheFirstFrameBeingProcessed{true};
    int m_frameCounter{0};
    depth_map_optimization::DepthMapOptimizationConfig m_depthMapOptimizationConfig;

    std::filesystem::path m_pathDepthMaps{""};
    std::filesystem::path m_pathOptimizationReports{""};
    bool m_doSaveDepthMaps{false};
    bool m_doSaveOptimizationReports{false};
    bool m_doRunRigorousOptimization{true};

    std::string stampToString(builtin_interfaces::msg::Time stamp) const;

public:

    DepthOptimizerNode();


};
