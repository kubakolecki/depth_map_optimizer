#pragma once

#include <ceres/ceres.h>

//parameters ordering: depth_ij, depth_kl, regression_slope

class DeltaDepthCostFunction : public ceres::SizedCostFunction<1,1,1,1>
{
    public:
        DeltaDepthCostFunction(double deltaDepth, double deltaDepthUncertainty): m_deltaDepth{deltaDepth}, m_deltaDepthUncertainty{deltaDepthUncertainty}
        {}

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobian) const override
        {
            const double predictedDeltaDepth = parameters[2][0]*(parameters[1][0] - parameters[0][0]);
            
            residuals[0] = predictedDeltaDepth - m_deltaDepth;
            residuals[0] /= m_deltaDepthUncertainty;

            if (jacobian != nullptr  && jacobian[0] != nullptr)
            {
                jacobian[0][0] = -parameters[2][0]/m_deltaDepthUncertainty; //derivative w.r.t depth_ij
                jacobian[1][0] = parameters[2][0]/m_deltaDepthUncertainty; //derivative w.r.t depth_kl
                jacobian[2][0] = (parameters[1][0] - parameters[0][0])/m_deltaDepthUncertainty; //derivative w.r.t regression_slope
            }

            return true;
        }


    private:
        double m_deltaDepth;
        double m_deltaDepthUncertainty;


};