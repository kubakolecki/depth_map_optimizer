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
        LossFunctionDescription ceresLossFunctionForDepthMap{TrivialLoss{}};
        LossFunctionDescription ceresLossFunctionForMapPoints{TrivialLoss{}};
        int scaleFactorForDepthMap{1};
        DepthMapOptimizationRoi roi{0,0,0,0};

    };
}