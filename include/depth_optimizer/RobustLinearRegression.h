#pragma once

#include <vector>


class RobustLinearRegression
{

//TODO implement calss RobustRegressionResult
//TODO implement enum class RobustRegressionStatus
//TODO return fit result as std::expected<RobustRegressionResult, RobustRegressionStatus> or something like this to be able to return error status in case of failure

public:
    RobustLinearRegression(float outlierThreshold, float outlierProbability);

    void fit(const std::vector<float>& x, const std::vector<float>& y);
    float predict(float x) const;

    int getNumberOfAttempts() const { return m_numberOfAttempts; }
    float getSlope() const { return m_slope; }
    float getIntercept() const { return m_intercept; }
    float getInlierRatio() const { return m_inlierRatio; }
    float getRmse() const { return m_rmse; }
    int getNumberOfInliers() const { return m_numberOfInliers; }


private:
    float m_slope{1.0f};
    float m_intercept{0.0f};
    float m_inlierRatio{0.0f};
    float m_rmse{0.0f};
    int m_numberOfInliers{0};


    
    float m_outlierThreshold{0.3f};
    float m_outlierProbability{0.5};
    int m_numberOfAttempts{1000};





};