#pragma once

#include <vector>
#include <expected>

struct RobustRegressionResult
{
    float slope{1.0f};
    float intercept{0.0f};
    float inlierRatio{0.0f};
    float rmse{0.0f};
    int numberOfInliers{0};
};

enum class RegressionFailureStatus
{
    NOT_ENOUGH_DATA,
    INLIER_RATIO_TOO_LOW,
    RMSE_TOO_HIGH,
    OTHER_ERROR
};

class RobustLinearRegression
{

//TODO implement calss RobustRegressionResult
//TODO implement enum class RobustRegressionStatus
//TODO return fit result as std::expected<RobustRegressionResult, RobustRegressionStatus> or something like this to be able to return error status in case of failure

public:
    RobustLinearRegression(float outlierThreshold, float outlierProbability);

    std::expected<RobustRegressionResult, RegressionFailureStatus> fit(const std::vector<float>& x, const std::vector<float>& y);
    float predict(float x) const;
    int getNumberOfAttempts() const { return m_numberOfAttempts; }

private:
    //Parameteres that take part in estimation:
    float m_outlierThreshold{0.3f};
    float m_outlierProbability{0.5};
    int m_numberOfAttempts{1000};

    //Limits for accepting the final reslut: TODO make them configurable via constructor or setter methods
    float m_inlierRatioLimit{0.7f};
    float m_rmseLimit{0.33f};





};