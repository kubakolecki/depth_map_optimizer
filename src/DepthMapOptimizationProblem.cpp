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


void DepthMapOptimizationProblem::fillOptimizationProblem(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints, const std::vector<float>& uncertainty)
{
    
    
    using clock = std::chrono::high_resolution_clock;
    auto timeStartT = clock::now();
    m_lossFunctionWrappersForMapPoints.clear();
    //m_lossFunctionWrappersForMapPoints.reserve(observedDepthMapPoints.size());

    const auto rowLimit {m_roiDecimated.rowMax - 1u};
    const auto colLimit {m_roiDecimated.colMax - 1u};
    const cv::Mat uncertaintyMap {m_config.depthMapUncertaintyCoefficient*m_depthMapDecimatedToOptimize};
    for (auto row = m_roiDecimated.rowMin; row < rowLimit; ++row) 
    {
        double* rowPtr = m_depthMapDecimatedToOptimize.ptr<double>(row);
        double* rowPtrNext = m_depthMapDecimatedToOptimize.ptr<double>(row+1);
        const double* rowPtrUncert = uncertaintyMap.ptr<double>(row);
        const double* rowPtrNextUncert = uncertaintyMap.ptr<double>(row+1);
        for (auto col = m_roiDecimated.colMin; col < colLimit; ++col) 
        {
            const auto deltaDepthCol {rowPtr[col+1] - rowPtr[col] };
            const auto deltaDepthRow {rowPtrNext[col] - rowPtr[col] };

            const auto deltaDepthColUncert {sqrt(rowPtrUncert[col+1]*rowPtrUncert[col+1] + rowPtrUncert[col]*rowPtrUncert[col])}; 
            const auto deltaDepthRowUncert {sqrt(rowPtrNextUncert[col]*rowPtrNextUncert[col] + rowPtrUncert[col]*rowPtrUncert[col])};

            ceres::CostFunction* deltaDepthCostFunctionCol = new DeltaDepthCostFunction(deltaDepthCol, deltaDepthColUncert);
            ceres::CostFunction* deltaDepthCostFunctionRow = new DeltaDepthCostFunction(deltaDepthRow, deltaDepthRowUncert);
            ceres::LossFunction* deltaDepthLossFunctionCol = this->createLossFunction(m_config.ceresLossFunctionForDepthMap);
            ceres::LossFunction* deltaDepthLossFunctionRow = this->createLossFunction(m_config.ceresLossFunctionForDepthMap);

            m_problem.AddResidualBlock(deltaDepthCostFunctionCol, deltaDepthLossFunctionCol, &rowPtr[col], &rowPtr[col+1], &m_slope);
            m_problem.AddResidualBlock(deltaDepthCostFunctionRow, deltaDepthLossFunctionRow, &rowPtr[col], &rowPtrNext[col], &m_slope);
        }
    }

    size_t pointIndex{0};
    for (const auto& point: observedDepthMapPoints)
    {
        
        if (point.y < m_config.roi.rowMin)
        {
            continue;
        }

        if (point.y >= m_config.roi.rowMax)
        {
            continue;
        }

        if (point.x < m_config.roi.colMin)
        {
            continue;
        }

        if (point.x >= m_config.roi.colMax)
        {
            continue;
        }

        const auto row = static_cast<unsigned int>(point.y/ static_cast<double>(m_config.scaleFactorForDepthMap));
        const auto col = static_cast<unsigned int>(point.x/ static_cast<double>(m_config.scaleFactorForDepthMap));

        const auto depthAtPoint = m_depthMapDecimatedToOptimize.at<double>(row, col);
        const auto depthDifference = std::abs(depthAtPoint - point.z);
        if (depthDifference > m_config.mapPointDifferenceThreshold)
        {
            std::cout << "Skipping point at (" << point.x << ", " << point.y << ") with depth " << point.z << " because the difference with the depth map value " << depthAtPoint << " is too large: " << depthDifference << std::endl; 
            continue;
        }
        
        ceres::CostFunction* depthCostFunction = new DepthCostFunction(point.z, uncertainty[pointIndex]);
        //ceres::LossFunction* lossFunction = this->createLossFunction(m_config.ceresLossFunctionForMapPoints);
        //auto lossFunctionWrapper =  std::make_shared<ceres::LossFunctionWrapper>(this->createLossFunction(m_config.ceresLossFunctionForMapPoints), ceres::Ownership::TAKE_OWNERSHIP);
        //m_lossFunctionWrappersForMapPoints.emplace_back(lossFunctionWrapper);

        //m_problem.AddResidualBlock(depthCostFunction, lossFunction, &m_depthMapDecimatedToOptimize.at<double>(row, col));

        //m_problem.AddResidualBlock(depthCostFunction, lossFunctionWrapper.get(), &m_depthMapDecimatedToOptimize.at<double>(row, col));

        //auto lossFncWrapper = ceres::LossFunctionWrapper(this->createLossFunction(m_config.ceresLossFunctionForMapPoints), ceres::Ownership::TAKE_OWNERSHIP);
        //m_lossFunctionWrappersForMapPoints.emplace_back(this->createLossFunction(m_config.ceresLossFunctionForMapPoints), ceres::Ownership::TAKE_OWNERSHIP);

        auto lossFunctionWrapperPtr = new ceres::LossFunctionWrapper(this->createLossFunction(m_config.ceresLossFunctionForMapPoints), ceres::Ownership::TAKE_OWNERSHIP);

        m_lossFunctionWrappersForMapPoints.emplace_back(lossFunctionWrapperPtr);

        m_problem.AddResidualBlock(depthCostFunction, lossFunctionWrapperPtr, &m_depthMapDecimatedToOptimize.at<double>(row, col));
        ++pointIndex;

    }

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    std::cout << "Filled optimization problem in " << durationT << " microseconds." << std::endl;

    
}

SolutionResult DepthMapOptimizationProblem::solve()
{
    using clock = std::chrono::high_resolution_clock;
    auto timeStartT = clock::now();
    ceres::Solver::Options options;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.linear_solver_type = ceres::CGNR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = m_config.numberOfCeresIterations;
    options.num_threads = 24;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &m_problem, &summary);
    std::cout << summary.FullReport() << "\n";

    options.max_num_iterations = m_config.numberOfCeresIterationsSecondStep;
    for (auto lossFunctionWrapperPtr : m_lossFunctionWrappersForMapPoints)
    {
        lossFunctionWrapperPtr->Reset(this->createLossFunction(m_config.ceresLossFunctionForMapPointsSecondStep), ceres::Ownership::TAKE_OWNERSHIP);
    }

    ceres::Solve(options, &m_problem, &summary);
    std::cout << summary.FullReport() << "\n";

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    std::cout << "Solved optimization problem in " << durationT << " microseconds." << std::endl;

    cv::Mat mapOfCorrections = m_depthMapDecimatedToOptimize - m_depthMapDecimatedOriginal;
    cv::Mat mapCorrectionsUpscaled;
    cv::resize(mapOfCorrections, mapCorrectionsUpscaled, m_depthMapOriginal.size(), 0, 0, cv::INTER_LINEAR);
    m_depthMapOriginal += mapCorrectionsUpscaled;

    const double sigmaZero {std::sqrt(2.0 * summary.final_cost / (summary.num_residuals - summary.num_effective_parameters))};
    const auto isSolutionUsable{summary.IsSolutionUsable()};

    return {sigmaZero, isSolutionUsable, summary.FullReport()};
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

DepthMapOptimizationProblem::DepthResiduals DepthMapOptimizationProblem::evaluateDepthResiduals(const std::vector<geometry_msgs::msg::Point32>& observedDepthMapPoints) const
{
    DepthMapOptimizationProblem::DepthResiduals depthResiduals;
    depthResiduals.reserve(observedDepthMapPoints.size());

    for (const auto& point: observedDepthMapPoints)
    {
        const auto row {static_cast<int>(point.y)};
        const auto col {static_cast<int>(point.x)};

        const auto residual{static_cast<float>(m_depthMapOriginal.at<double>(row, col) - point.z)};
        depthResiduals.emplace_back(point.x, point.y, residual);
    }

    std::sort(depthResiduals.begin(), depthResiduals.end(), [](const auto& a, const auto& b)
    {
        return std::abs(a[2]) > std::abs(b[2]);
    });

    return depthResiduals;
}

