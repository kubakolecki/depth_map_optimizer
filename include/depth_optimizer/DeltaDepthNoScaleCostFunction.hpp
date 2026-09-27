#pragma once

#include <ceres/ceres.h>

//parameters ordering: depth_ij, depth_kl

class DeltaDepthNoScaleCostFunction : public ceres::SizedCostFunction<1,1,1>
{
    public:
        DeltaDepthNoScaleCostFunction(double deltaDepth, double deltaDepthUncertainty): m_deltaDepth{deltaDepth}, m_deltaDepthUncertainty{deltaDepthUncertainty}
        {}

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobian) const override
        {
            const double predictedDeltaDepth = parameters[1][0] - parameters[0][0];
            
            residuals[0] = predictedDeltaDepth - m_deltaDepth;
            residuals[0] /= m_deltaDepthUncertainty;

            if (jacobian != nullptr  && jacobian[0] != nullptr)
            {
                jacobian[0][0] = -1.0/m_deltaDepthUncertainty; //derivative w.r.t depth_ij
                jacobian[1][0] = 1.0/m_deltaDepthUncertainty; //derivative w.r.t depth_kl
            }

            return true;
        }

    private:
        double m_deltaDepth;
        double m_deltaDepthUncertainty;
};