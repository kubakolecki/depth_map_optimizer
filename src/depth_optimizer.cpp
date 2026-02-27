#include "depth_optimizer/DepthOptimizerNode.hpp"


int main(int argc, char * argv[] )
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthOptimizerNode>());
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}