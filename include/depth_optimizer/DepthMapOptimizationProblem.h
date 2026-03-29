#pragma once

#include <geometry_msgs/msg/point32.hpp>

#include <opencv2/opencv.hpp>
#include <ceres/problem.h>


#include <vector>

struct DepthMapOptimizationRoi
{
    unsigned int rowMin;
    unsigned int rowMax;
    unsigned int colMin;
    unsigned int colMax;
};


class DepthMapOptimizationProblem
{
    public:
        explicit DepthMapOptimizationProblem(cv::Mat& depthMap,  const DepthMapOptimizationRoi roi, double slope, int scaleFactor);
        void fillOptimizationProblem(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints);
        void solve();
        double getSlope() const { return m_slope; }


    private:
        cv::Mat& m_depthMapOriginal;
        cv::Mat m_depthMapDecimated; //This is the depth map that will be optimized,
        DepthMapOptimizationRoi m_roi;
        DepthMapOptimizationRoi m_roiDecimated;
        ceres::Problem m_problem;
        double m_slope{1.0};
        int m_scaleFactor{1};
};