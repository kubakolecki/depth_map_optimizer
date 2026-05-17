#include "depth_optimizer/io.hpp"

#include <iomanip>

void depth_optmizer_io::sendToStream(std::ostream &outputStream, const RobustRegressionResult& regressionResult)
{
    outputStream << "Robust linear regression result:\n";
    outputStream << std::fixed << std::setprecision(12);
    outputStream << "Slope: " << regressionResult.slope << "\n";
    outputStream << "Intercept: " << regressionResult.intercept << "\n";
    outputStream << "Inlier_ratio: " << regressionResult.inlierRatio << "\n";
    outputStream << "RMSE: " << regressionResult.rmse << "\n";
    outputStream << "Number_of_inliers: " << regressionResult.numberOfInliers << "\n";
    outputStream << "Number_of_outliers: " << regressionResult.numberOfOutliers << "\n";
}

void depth_optmizer_io::sendToStream(std::ostream &outputStream, const depth_map_optimization::DepthMapOptimizationProblem::DepthResiduals& depthResiduals)
{
    outputStream << "Residuals of observed depth:\n";
    outputStream << "x,y,delta_depth\n";
    outputStream << std::fixed << std::setprecision(5);
    for (const auto& residual : depthResiduals)
    {
        outputStream << residual(0) << "," << residual(1) << "," << residual(2) << "\n";
    }
}

void depth_optmizer_io::sendToStream(std::ostream &outputStream, const depth_map_optimization::SolutionResult& solutionResult)
{
    
    outputStream << "Is solution usable: " << (solutionResult.isSolutionUsable ? "Yes" : "No") << "\n";
    outputStream << "Sigma zero: " << solutionResult.sigmaZero << "\n";
    outputStream << "Ceres Solver report:\n";
    outputStream << solutionResult.solverReport << "\n";

}