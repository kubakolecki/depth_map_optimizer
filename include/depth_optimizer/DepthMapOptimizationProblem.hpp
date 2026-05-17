#pragma once

#include "DepthMapOptimizationConfig.hpp"

#include <geometry_msgs/msg/point32.hpp>

#include <opencv2/opencv.hpp>
#include <ceres/problem.h>


#include <vector>


namespace depth_map_optimization
{
struct SolutionResult
{
    double sigmaZero{1.0};
    bool isSolutionUsable{true};
    std::string solverReport{};
};

class DepthMapOptimizationProblem
{
    public:
        using DepthResidual = cv::Vec3f;
        using DepthResiduals = std::vector<DepthResidual>;
        
        explicit DepthMapOptimizationProblem(cv::Mat& depthMap,  double slope, const DepthMapOptimizationConfig& config);
        void fillOptimizationProblem(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints);
        SolutionResult solve();
        double getSlope() const { return m_slope; }
        DepthResiduals evaluateDepthResiduals(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints) const;


    private:

        template <class... Ts>
        struct OverloadLossFunctionCreator : Ts...
        {
            using Ts::operator()...;
        };

        ceres::LossFunction* createLossFunction(const LossFunctionDescription& lossFunctionDescription) const;

        cv::Mat& m_depthMapOriginal;
        double m_slope{1.0};
        DepthMapOptimizationConfig m_config;
        
        cv::Mat m_depthMapDecimatedToOptimize; //This is the depth map after decimation that will be optimized
        cv::Mat m_depthMapDecimatedOriginal; //This is the depth map after decimation that will not be modified and will only serve for calculating corrections
        DepthMapOptimizationRoi m_roiDecimated;
        ceres::Problem m_problem;
        
};

}