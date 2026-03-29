#pragma once

#include <ceres/ceres.h>

//parameter ordering: depth_ij

class DepthCostFunction : public ceres::SizedCostFunction<1,1>
{
    public:
        DepthCostFunction(double depth, double depthUncertainty): m_depth(depth), m_depthUncertainty(depthUncertainty)
        {}

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobian) const override
        {

            residuals[0] = parameters[0][0] - m_depth;
            residuals[0] /= m_depthUncertainty;

            if (jacobian != nullptr  && jacobian[0] != nullptr)
            {
                 jacobian[0][0] = 1.0/m_depthUncertainty;
            }


            return true;
        }



    private:
        double m_depth;
        double m_depthUncertainty;

};