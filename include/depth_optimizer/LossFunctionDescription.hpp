#pragma once

#include <variant>

namespace depth_map_optimization
{
struct TrivialLoss
{
};

struct CauchyLoss
{
    double parameter{3.0};
};

struct HuberLoss
{
    double parameter{2.0};
};

struct TukeyLoss
{
    double parameter{2.0};
};

using LossFunctionDescription = std::variant<TrivialLoss, CauchyLoss, HuberLoss, TukeyLoss>;

}