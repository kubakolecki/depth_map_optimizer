#include "depth_optimizer/DepthMapOptimizationProblem.hpp"
#include "depth_optimizer/DeltaDepthCostFunction.hpp"
#include "depth_optimizer/DepthCostFunction.hpp"

#include <iostream>
#include <chrono>


using namespace depth_map_optimization;

DepthMapOptimizationProblem::DepthMapOptimizationProblem(cv::Mat& depthMapOriginal, double slope, const DepthMapOptimizationConfig& config): 
m_depthMapOriginal(depthMapOriginal),
m_slope(slope),
m_config(config),
m_roiDecimated{config.roi.rowMin/config.scaleFactorForDepthMap, config.roi.rowMax/config.scaleFactorForDepthMap, config.roi.colMin/config.scaleFactorForDepthMap, config.roi.colMax/config.scaleFactorForDepthMap}
{
    cv::resize(m_depthMapOriginal, m_depthMapDecimatedToOptimize, cv::Size(), 1.0/static_cast<double>(m_config.scaleFactorForDepthMap), 1.0/static_cast<double>(m_config.scaleFactorForDepthMap), cv::INTER_NEAREST);
    m_depthMapDecimatedOriginal = m_depthMapDecimatedToOptimize.clone();
}


void DepthMapOptimizationProblem::fillOptimizationProblem(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints)
{
    
    
    using clock = std::chrono::high_resolution_clock;
    auto timeStartT = clock::now();
    const auto rowLimit {m_roiDecimated.rowMax - 1u};
    const auto colLimit {m_roiDecimated.colMax - 1u};
    for (auto row = m_roiDecimated.rowMin; row < rowLimit; ++row) 
    {
        double* rowPtr = m_depthMapDecimatedToOptimize.ptr<double>(row);
        double* rowPtrNext = m_depthMapDecimatedToOptimize.ptr<double>(row+1);
        for (auto col = m_roiDecimated.colMin; col < colLimit; ++col) 
        {
            const auto deltaDepthCol {rowPtr[col+1] - rowPtr[col] };
            const auto deltaDepthRow {rowPtrNext[col] - rowPtr[col] };

            ceres::CostFunction* deltaDepthCostFunctionCol = new DeltaDepthCostFunction(deltaDepthCol, 0.1);
            ceres::CostFunction* deltaDepthCostFunctionRow = new DeltaDepthCostFunction(deltaDepthRow, 0.1);
            ceres::LossFunction* deltaDepthLossFunctionCol = this->createLossFunction(m_config.ceresLossFunctionForDepthMap);
            ceres::LossFunction* deltaDepthLossFunctionRow = this->createLossFunction(m_config.ceresLossFunctionForDepthMap);

            m_problem.AddResidualBlock(deltaDepthCostFunctionCol, deltaDepthLossFunctionCol, &rowPtr[col], &rowPtr[col+1], &m_slope);
            m_problem.AddResidualBlock(deltaDepthCostFunctionRow, deltaDepthLossFunctionRow, &rowPtr[col], &rowPtrNext[col], &m_slope);
        }
    }


    for (const auto& point: observedDepthMapPoints)
    {
        
        if (point.y < m_config.roi.rowMin || point.y >= m_config.roi.rowMax || point.x < m_config.roi.colMin || point.x >= m_config.roi.colMax)
        {
            continue;
        }
        
        const auto row = static_cast<unsigned int>(point.y/ static_cast<double>(m_config.scaleFactorForDepthMap));
        const auto col = static_cast<unsigned int>(point.x/ static_cast<double>(m_config.scaleFactorForDepthMap));

        ceres::CostFunction* depthCostFunction = new DepthCostFunction(point.z, 0.05 * point.z);
        ceres::LossFunction* lossFunction = this->createLossFunction(m_config.ceresLossFunctionForMapPoints);
        m_problem.AddResidualBlock(depthCostFunction, lossFunction, &m_depthMapDecimatedToOptimize.at<double>(row, col));
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
    options.max_num_iterations = m_config.numberOfCeresIterations;
    options.num_threads = 24;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &m_problem, &summary);
    std::cout << summary.FullReport() << "\n";

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    std::cout << "Solved optimization problem in " << durationT << " microseconds." << std::endl;

    cv::Mat mapOfCorrections = m_depthMapDecimatedToOptimize - m_depthMapDecimatedOriginal;
    cv::Mat mapCorrectionsUpscaled;
    cv::resize(mapOfCorrections, mapCorrectionsUpscaled, m_depthMapOriginal.size(), 0, 0, cv::INTER_LINEAR);
    m_depthMapOriginal += mapCorrectionsUpscaled;


}

ceres::LossFunction* DepthMapOptimizationProblem::createLossFunction(const LossFunctionDescription& lossFunctionDescription) const
{
    ceres::LossFunction *lossFcnPtr = std::visit(OverloadLossFunctionCreator{
                                                     []([[maybe_unused]] TrivialLoss loss) {
                                                         ceres::LossFunction *lFPtr = new ceres::TrivialLoss{};
                                                         return lFPtr;
                                                     },
                                                     []([[maybe_unused]] CauchyLoss loss) {
                                                         ceres::LossFunction *lFPtr = new ceres::CauchyLoss{loss.parameter};
                                                         return lFPtr;
                                                     },
                                                     []([[maybe_unused]] HuberLoss loss) {
                                                         ceres::LossFunction *lFPtr = new ceres::HuberLoss{loss.parameter};
                                                         return lFPtr;
                                                     },
                                                     []([[maybe_unused]] TukeyLoss loss) {
                                                         ceres::LossFunction *lFPtr = new ceres::TukeyLoss{loss.parameter};
                                                         return lFPtr;
                                                     },
                                                 },
                                                 lossFunctionDescription);
    return lossFcnPtr;

}

