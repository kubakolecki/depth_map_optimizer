#include "depth_optimizer/DepthMapOptimizationProblem.h"
#include "depth_optimizer/DeltaDepthCostFunction.hpp"
#include "depth_optimizer/DepthCostFunction.hpp"

#include <iostream>
#include <chrono>

DepthMapOptimizationProblem::DepthMapOptimizationProblem(cv::Mat& depthMapOriginal, const DepthMapOptimizationRoi roi, double slope, int scaleFactor): 
m_depthMapOriginal(depthMapOriginal),
m_roi(roi),
m_roiDecimated({roi.rowMin/scaleFactor, roi.rowMax/scaleFactor, roi.colMin/scaleFactor, roi.colMax/scaleFactor}),
m_slope(slope),
m_scaleFactor(scaleFactor)
{
    cv::resize(m_depthMapOriginal, m_depthMapDecimated, cv::Size(), 1.0/static_cast<double>(scaleFactor), 1.0/static_cast<double>(scaleFactor), cv::INTER_NEAREST);
}


void DepthMapOptimizationProblem::fillOptimizationProblem(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints)
{
    using clock = std::chrono::high_resolution_clock;
    

    auto timeStartT = clock::now();
    const auto rowLimit {m_roiDecimated.rowMax - 1u};
    const auto colLimit {m_roiDecimated.colMax - 1u};
    for (auto row = m_roiDecimated.rowMin; row < rowLimit; ++row) 
    {
        double* rowPtr = m_depthMapDecimated.ptr<double>(row);
        double* rowPtrNext = m_depthMapDecimated.ptr<double>(row+1);
        for (auto col = m_roiDecimated.colMin; col < colLimit; ++col) 
        {
            const auto deltaDepthCol {rowPtr[col+1] - rowPtr[col] };
            const auto deltaDepthRow {rowPtrNext[col] - rowPtr[col] };

            ceres::CostFunction* deltaDepthCostFunctionCol = new DeltaDepthCostFunction(deltaDepthCol, 0.3);
            ceres::CostFunction* deltaDepthCostFunctionRow = new DeltaDepthCostFunction(deltaDepthRow, 0.3);

            ceres::LossFunction* lossFunctionCol{nullptr};
            ceres::LossFunction* lossFunctionRow{nullptr};

            m_problem.AddResidualBlock(deltaDepthCostFunctionCol, lossFunctionCol, &rowPtr[col], &rowPtr[col+1], &m_slope);
            m_problem.AddResidualBlock(deltaDepthCostFunctionRow, lossFunctionRow, &rowPtr[col], &rowPtrNext[col], &m_slope);
        }
    }


    for (const auto& point: observedDepthMapPoints)
    {
        
        if (point.y < m_roi.rowMin || point.y >= m_roi.rowMax || point.x < m_roi.colMin || point.x >= m_roi.colMax)
        {
            continue;
        }
        
        const auto row = static_cast<unsigned int>(point.y/ static_cast<double>(m_scaleFactor));
        const auto col = static_cast<unsigned int>(point.x/ static_cast<double>(m_scaleFactor));

        ceres::CostFunction* depthCostFunction = new DepthCostFunction(point.z, 0.05);
        ceres::LossFunction* lossFunction{nullptr};
        m_problem.AddResidualBlock(depthCostFunction, lossFunction, &m_depthMapDecimated.at<double>(row, col));
    }

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    std::cout << "Filled optimization problem in " << durationT << " microseconds." << std::endl;

}

void DepthMapOptimizationProblem::solve()
{
    using clock = std::chrono::high_resolution_clock;
    auto timeStartT = clock::now();
    ceres::Solver::Options options;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.linear_solver_type = ceres::CGNR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 4;
    options.num_threads = 24;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &m_problem, &summary);
    std::cout << summary.FullReport() << "\n";

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    std::cout << "Solved optimization problem in " << durationT << " microseconds." << std::endl;

}

