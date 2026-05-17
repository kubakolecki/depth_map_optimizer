#pragma once

#include "depth_optimizer/RobustLinearRegression.hpp"
#include "depth_optimizer/DepthMapOptimizationProblem.hpp"

#include <ostream>

namespace depth_optmizer_io
{

void sendToStream(std::ostream &outputStream, const RobustRegressionResult& regressionResult);
void sendToStream(std::ostream &outputStream, const depth_map_optimization::DepthMapOptimizationProblem::DepthResiduals& depthResiduals);
void sendToStream(std::ostream &outputStream, const depth_map_optimization::SolutionResult& solutionResult);


}
