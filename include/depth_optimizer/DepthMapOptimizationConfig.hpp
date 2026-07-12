#pragma once


#include "LossFunctionDescription.hpp"

namespace depth_map_optimization
{
    struct DepthMapOptimizationRoi
    {
        unsigned int rowMin;
        unsigned int rowMax;
        unsigned int colMin;
        unsigned int colMax;
    };
    
    
    struct DepthMapOptimizationConfig
    {   
        int numberOfCeresIterations{4};
        int numberOfCeresIterationsSecondStep{2};
        float mapPointDifferenceThreshold{0.5f};
        LossFunctionDescription ceresLossFunctionForDepthMap{TrivialLoss{}};
        LossFunctionDescription ceresLossFunctionForMapPoints{TrivialLoss{}};
        LossFunctionDescription ceresLossFunctionForMapPointsSecondStep{TrivialLoss{}};
        int scaleFactorForDepthMap{1};
        DepthMapOptimizationRoi roi{0,0,0,0};

    };
}