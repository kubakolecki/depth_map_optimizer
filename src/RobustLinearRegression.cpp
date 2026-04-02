#include "depth_optimizer/RobustLinearRegression.hpp"


#include <numeric>
#include <ranges>
#include <random>
#include <algorithm>

#include <iostream>
#include <chrono>

RobustLinearRegression::RobustLinearRegression(float outlierThreshold, float outlierProbability): m_outlierThreshold{outlierThreshold}, m_outlierProbability{outlierProbability}
{
    m_numberOfAttempts =  static_cast<int>(std::log(1.0f - 0.999f) / std::log(1.0f - std::pow(1.0f - m_outlierProbability, 2.0f)));


    std::cout << "Number of RANSAC attempts set to: " << m_numberOfAttempts << "\n";
}

std::expected<RobustRegressionResult, RegressionFailureStatus> RobustLinearRegression::fit(const std::vector<float>& x, const std::vector<float>& y)
{
    using clock = std::chrono::high_resolution_clock;
    auto timeStartRansac = clock::now();
    
    using SlopeType = float;
    using InterceptType = float;
    using SampleId = size_t;
    using InlierId = SampleId;
    using Sample = std::pair<SampleId, SampleId>;
    using NumberOfInliersType = size_t;
    using SlopeAndIntercept = std::pair<SlopeType, InterceptType>;
    using BestSampleResult = std::tuple<SampleId, NumberOfInliersType, std::vector<InlierId>>;


    const auto numberOfPoints{x.size()};

    if (numberOfPoints < 2)
    {
        return std::unexpected{RegressionFailureStatus::NOT_ENOUGH_DATA};
    }

    std::uniform_int_distribution<size_t> distribution{0, numberOfPoints-1};
    std::random_device device;
    std::mt19937 engine {device()};
   

    std::vector<Sample> samples(m_numberOfAttempts, Sample{0,0});
    std::ranges::generate(samples, [&engine, &distribution](){ return std::make_pair(distribution(engine),distribution(engine)); });
    
    auto comuteRegressionForMinimalSample = [&x, &y](const auto& indices)
    {
        const auto[id1, id2] = indices;
        if (id1 == id2)
        {
            return std::make_pair(99.0f,0.0f);
        }

        const auto slope = (y[id2] - y[id1])/(x[id2] - x[id1]);
        const auto intercept = y[id1] - slope*x[id1];
        return std::make_pair(slope, intercept);
    };
    
    std::vector<SlopeAndIntercept> regressionResults(m_numberOfAttempts, SlopeAndIntercept{});
    std::ranges::transform(samples, regressionResults.begin(), comuteRegressionForMinimalSample);

    const auto outlierThreshold{m_outlierThreshold};


    auto getInliers = [&x, &y, outlierThreshold](const SlopeAndIntercept& regressionResult)
    {
        const auto numberOfSamples{x.size()};
        
        const auto& [slope, intercept]{regressionResult};

        std::vector<float> residuals;
        residuals.reserve(numberOfSamples);

        std::ranges::transform(x, y, std::back_inserter(residuals), [slope = slope, intercept = intercept](const auto x, const auto y){
            const auto yPredicted = slope*x + intercept;
            const auto residual = y - yPredicted;
            return residual;
          });
    
        std::vector<InlierId> inlierIndices;
        inlierIndices.reserve(numberOfSamples);
        for (size_t i = 0; i<numberOfSamples; ++i)
        {
            if ( std::fabs(residuals[i]) < outlierThreshold)
            {
                inlierIndices.push_back(i);
            }
        }
        inlierIndices.shrink_to_fit();
        return inlierIndices;
    };


    BestSampleResult bestResult(0, 0, std::vector<InlierId>() );
    auto numberOfGoodSmaplsFound{0};
    for (auto sampleId = 0; sampleId<m_numberOfAttempts; ++sampleId)
    {
        auto& [bestSampleIdSoFar, highestNumberOfInliersSoFar, inliersForBestSampleSoFar]{bestResult};
        const auto inliers { getInliers(regressionResults[sampleId]) };
        const auto inlierRatio = static_cast<float>(inliers.size())/ static_cast<float>(numberOfPoints);

        if (inliers.size() > highestNumberOfInliersSoFar)
        {
            bestSampleIdSoFar = sampleId;
            highestNumberOfInliersSoFar = inliers.size();
            inliersForBestSampleSoFar = inliers;
        }


        if (inlierRatio > (1.0f - m_outlierProbability)) //We consider it a good sample
        {
            numberOfGoodSmaplsFound++;
        }

        if (numberOfGoodSmaplsFound > 20) //To add robustness we assume we need several good samples
        {
            break;
        }
    }

    auto timeEndRansac = clock::now();
    auto durationRansac = std::chrono::duration_cast<std::chrono::microseconds>(timeEndRansac - timeStartRansac).count();
    std::cout << "RANSAC fitting took " << durationRansac << " microseconds\n";


    const auto& [bestSampleId, numberOfInliers, inlierIndices]{bestResult};
    const auto inlierRatio {static_cast<float>(numberOfInliers)/static_cast<float>(numberOfPoints)};

    if (inlierRatio < m_inlierRatioLimit)
    {
        return std::unexpected{RegressionFailureStatus::INLIER_RATIO_TOO_LOW};
    }

    const auto [bestSlopeSoFar, bestInterceptSoFar]{regressionResults.at(bestSampleId)};
    std::vector<float> xInliers;
    std::vector<float> yInliers;
    xInliers.reserve(numberOfInliers);
    yInliers.reserve(numberOfInliers);
    for (const auto& inlierId : inlierIndices)
    {
        xInliers.push_back(x[inlierId]);
        yInliers.push_back(y[inlierId]);
    }

    //final fit using all inliers:

    const auto sumXSquared = std::transform_reduce(xInliers.begin(), xInliers.end(), 0.0f, std::plus{}, [](const auto val){return val*val;});
    const auto sumX = std::reduce(xInliers.begin(), xInliers.end(), 0.0f);
    const auto sumY = std::reduce(yInliers.begin(), yInliers.end(), 0.0f);
    const auto sumXY = std::transform_reduce(xInliers.begin(), xInliers.end(), yInliers.begin(), 0.0f, std::plus{}, [](const auto xVal, const auto yVal){ return xVal * yVal; } );
    const auto denominator = numberOfInliers * sumXSquared - sumX * sumX;
    const auto intercept = (sumY*sumXSquared - sumX*sumXY)/denominator;
    const auto slope = (numberOfInliers*sumXY - sumX*sumY)/denominator;

    std::vector<float> residuals;
    residuals.reserve(numberOfInliers);
    std::ranges::transform(xInliers, yInliers, std::back_inserter(residuals), [slope = slope, intercept = intercept](const auto x, const auto y){
        const auto yPredicted{slope*x + intercept};
        const auto residual{y - yPredicted};
        return residual;
      });

    const auto sumOfSquaredErrors = std::transform_reduce(residuals.begin(), residuals.end(), 0.0f, std::plus{}, [](auto val){return val*val;});
    const auto rmse = std::sqrt( sumOfSquaredErrors / static_cast<float>(numberOfInliers) );
    
    if (rmse > m_rmseLimit)
    {
        return std::unexpected{RegressionFailureStatus::RMSE_TOO_HIGH};
    }
        RobustRegressionResult finalResult{slope, intercept, inlierRatio, rmse, static_cast<int>(numberOfInliers)};

    return finalResult;
    
    //for debugin only: TODO: remove when not needed anymore
    /*
    std::vector<float> residualsAll;
    residualsAll.reserve(numberOfPoints);
    std::ranges::transform(x, y, std::back_inserter(residualsAll), [slope = m_slope, intercept = m_intercept](const auto x, const auto y){
        const auto yPredicted = slope*x + intercept;
        const auto residual = std::fabs(y - yPredicted);
        return residual;
        });

    std::sort(residualsAll.begin(), residualsAll.end());
    std::cout << "Median residual: " << residualsAll[residualsAll.size()/2] << "\n";
    std::cout << "90th percentile residual: " << residualsAll[static_cast<size_t>(residualsAll.size()*0.9f)] << "\n";
    std::cout << "80th percentile residual: " << residualsAll[static_cast<size_t>(residualsAll.size()*0.8f)] << "\n";
    std::cout << "number of inliers: " << numberOfInliers << " out of " << numberOfPoints << " points\n";
    */




    //printing samples for debugging
    //for (const auto& sample : samples)
    //{
    //    std::cout << "Sample: (" << sample.first << ", " << sample.second << ")\n";
    //}





};