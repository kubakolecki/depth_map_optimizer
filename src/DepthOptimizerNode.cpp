#include "depth_optimizer/DepthOptimizerNode.hpp"
#include "depth_optimizer/msg/object_data.hpp"
#include "depth_optimizer/DepthMapOptimizationProblem.h"

#include <geometry_msgs/msg/point32.hpp>


#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/parallel/backend/parallel_for.tbb.hpp>


#include <Eigen/Dense>

#include <functional>
#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream> //TODO remove this after debugging
#include <fstream> //TODO remove this after debugging

DepthOptimizerNode::DepthOptimizerNode(): Node("depth_optimizer_node")
{
    auto paramObjectDataTopicNameDescription{rcl_interfaces::msg::ParameterDescriptor{}};
    auto paramNumberOfThreadsDescription{rcl_interfaces::msg::ParameterDescriptor{}};
    paramObjectDataTopicNameDescription.description = "topic name for object data message";
    paramNumberOfThreadsDescription.description = "number of threads for OpenCV operations";
    this->declare_parameter<std::string>("object_data_topic_name","/slam_deep_mapper/object_data", paramObjectDataTopicNameDescription);
    this->declare_parameter<int>("opencv_number_of_threads",16, paramNumberOfThreadsDescription);
    
    objectDataSubscriber = this->create_subscription<depth_optimizer::msg::ObjectData>(this->get_parameter("object_data_topic_name").as_string(),10,
        std::bind(&DepthOptimizerNode::objectDataCallback, this, std::placeholders::_1)
    );

    //TODO: looks like this is not needed to be set, OpenCV uses all available threads by default, remove later
    //cv::setNumThreads(this->get_parameter("opencv_number_of_threads").as_int()); //TODO test if this works, also test this in other nodes

    RCLCPP_INFO(this->get_logger(), "The number of opencv threads has been set to %d", cv::getNumThreads());
    cv::parallel::setParallelForBackend(std::make_shared<cv::parallel::tbb::ParallelForBackend>());

    const auto outlierThreshold{0.2f}; 
    const auto outlierProbability{0.5f}; 

    m_robustLinearRegression = std::make_unique<RobustLinearRegression>(outlierThreshold, outlierProbability);

}


void DepthOptimizerNode::objectDataCallback(const depth_optimizer::msg::ObjectData::SharedPtr msg)
{
    using clock = std::chrono::high_resolution_clock;
    RCLCPP_INFO(this->get_logger(), "Received ObjectData message with: %d objects", msg->number_of_objects);


    std::string pathFilePointCloudForDebug = "pointcloud_" + std::to_string(m_frameCounter) + "_" + std::to_string(msg->number_of_objects) + ".txt";

    //getting depth values from depth map for map points with corresponding 2D points in image

    //const auto numberOfMapPoints{msg->sparse_depth_information.points.size()};

    //TODO: probably there is no need to interpolate depth values for depth map, there would be no significant accuracy improvement.
    //TODO: implement direct retrieval of depth values from depth map.

    auto isInValidRange = [&msg](const auto& point)
    {
        return point.x >= msg->depth_map_col_min && point.x < msg->depth_map_col_max && point.y >= msg->depth_map_row_min && point.y < msg->depth_map_row_max;
    };

    auto timeStartT = clock::now();

    auto coordinatesX = msg->sparse_depth_information.points | std::views::filter(isInValidRange) | std::views::transform([](const auto& point){return point.x;}) | std::ranges::to<std::vector<float>>();
    auto coordinatesY = msg->sparse_depth_information.points | std::views::filter(isInValidRange) | std::views::transform([](const auto& point){return point.y;}) | std::ranges::to<std::vector<float>>();
    auto depthValuesFromSparseMap = msg->sparse_depth_information.points | std::views::filter(isInValidRange) | std::views::transform([](const auto& point){return point.z;}) | std::ranges::to<std::vector<float>>();

    const auto numberOfValidMapPoints{depthValuesFromSparseMap.size()};

    auto timeEndT = clock::now();
    auto durationT = std::chrono::duration_cast<std::chrono::microseconds>(timeEndT - timeStartT).count();
    RCLCPP_INFO(this->get_logger(), "Transformed sparse depth information points in %ld microseconds", durationT);

    cv::Mat depthMap(msg->rows, msg->columns, CV_32FC1, msg->depth_map_left_row_major.data());

    cv::Mat mapX(1, numberOfValidMapPoints, CV_32FC1, coordinatesX.data());
    cv::Mat mapY(1, numberOfValidMapPoints, CV_32FC1, coordinatesY.data());

    std::vector<float> depthValuesFromDepthMap(numberOfValidMapPoints,0.0f);
    cv::Mat depthValuesWrapper(1, numberOfValidMapPoints, CV_32FC1, depthValuesFromDepthMap.data());

    auto timeStartRemap = clock::now();
    cv::remap(depthMap, depthValuesWrapper, mapX, mapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0.0f);
    auto timeEndRemap = clock::now();
    auto durationRemap = std::chrono::duration_cast<std::chrono::microseconds>(timeEndRemap - timeStartRemap).count();
    RCLCPP_INFO(this->get_logger(), "Retrieved depth values from depth map in %ld microseconds", durationRemap);

    //std::for_each(depthValuesFromDepthMap.begin(), depthValuesFromDepthMap.end(), [](const auto& depthValue){
    //    RCLCPP_INFO(rclcpp::get_logger("DepthOptimizerNode"), "Depth value: %f", depthValue);
    //});

    //for (auto pointIdx = 0ull; pointIdx < numberOfMapPoints; ++pointIdx)
    //{
    //    const auto depthValueFromSparseMap = msg->sparse_depth_information.points[pointIdx].z;
    //    const auto depthValueFromDepthMap = depthValuesFromDepthMap[pointIdx];
    //    //RCLCPP_INFO(this->get_logger(), "Point idx: %lld, depth from sparse map: %f, depth from depth map: %f", pointIdx, depthValueFromSparseMap, depthValueFromDepthMap);
    //}

    m_robustLinearRegression->fit(depthValuesFromDepthMap, depthValuesFromSparseMap);

    RCLCPP_INFO(this->get_logger(), "Completed robust linear regression fitting.");
    RCLCPP_INFO(this->get_logger(), "Inlier ratio: %f, Number of inliers: %d, Slope: %f, Intercept: %f, RMSE: %f", 
        m_robustLinearRegression->getInlierRatio(),
        m_robustLinearRegression->getNumberOfInliers(),
        m_robustLinearRegression->getSlope(),
        m_robustLinearRegression->getIntercept(),
        m_robustLinearRegression->getRmse()
    );

    if (m_robustLinearRegression->getRmse() > 0.5f) //TODO find a good threshold for rmse to decide when to apply linear correction, also test if this is needed at all, maybe we can just always apply linear correction
    {
        RCLCPP_WARN(this->get_logger(), "High RMSE for robust linear regression fit: %f. Consider not applying linear correction to depth map.", m_robustLinearRegression->getRmse());
        RCLCPP_WARN(this->get_logger(), "Depth values from sparse map and depth map might be significantly different");  
        return;
    }

    cv::Mat depthMapToOptmize;
    depthMap.convertTo(depthMapToOptmize, CV_64F);
    
    DepthMapOptimizationProblem depthMapOptimizationProblem(
        depthMapToOptmize,
        DepthMapOptimizationRoi{msg->depth_map_row_min, msg->depth_map_row_max, msg->depth_map_col_min, msg->depth_map_col_max},
        static_cast<double>(1.0/m_robustLinearRegression->getSlope()),
        4
    );


    depthMapOptimizationProblem.fillOptimizationProblem(msg->sparse_depth_information.points);
    depthMapOptimizationProblem.solve();

    RCLCPP_INFO(this->get_logger(), "Slope after optimization: %f", 1.0/depthMapOptimizationProblem.getSlope());

    cv::Mat depthMapAfterLinearCorrection = depthMap * m_robustLinearRegression->getSlope() + m_robustLinearRegression->getIntercept();
 
    auto timeStartTransformation = clock::now();

    cv::Mat imageSegmentedByObjectIds (msg->rows, msg->columns, CV_16UC1, msg->image_segmented_by_classes.data() ); //Initialize with zeros

    const auto numberOfObjects{msg->number_of_objects};

    for (auto classId: msg->list_of_classes)
    {
        RCLCPP_INFO(this->get_logger(), "Detected object with class id %d", classId);
    }


    if (m_isTheFirstFrameBeingProcessed)
    {
        cv::Mat yRange(msg->rows, 1, CV_32F);
        for (unsigned int i = 0; i < msg->rows; ++i)
        {
            yRange.at<float>(i, 0) = static_cast<float>(i);
        }

        cv::Mat xRange(1, msg->columns, CV_32F);
        for (unsigned int j = 0; j < msg->columns; ++j)
        {
            xRange.at<float>(0, j) = static_cast<float>(j);
        }

        m_imageCoordinatesY = cv::repeat(yRange, 1, msg->columns) - static_cast<float>(msg->rows)/2.0f + 0.5f;
        m_imageCoordinatesX = cv::repeat(xRange, msg->rows, 1) - static_cast<float>(msg->columns)/2.0f + 0.5f;
        m_ones = cv::Mat::ones(msg->rows, msg->columns, CV_32F);
    }

    //RCLCPP_INFO(this->get_logger(), "Focal length for left camera is: %f %f", msg->focal_lenght_left[0], msg->focal_lenght_left[1]);

    cv::Mat cameraFrameCoordinatesX;
    cv::Mat cameraFrameCoordinatesY;
    cv::multiply(m_imageCoordinatesX, depthMapAfterLinearCorrection, cameraFrameCoordinatesX, (1.0f / msg->focal_lenght_left[0]));
    cv::multiply(m_imageCoordinatesY, depthMapAfterLinearCorrection, cameraFrameCoordinatesY, (1.0f / msg->focal_lenght_left[1]));

    CV_Assert(m_imageCoordinatesX.size() == depthMapAfterLinearCorrection.size() && m_imageCoordinatesY.size() == depthMapAfterLinearCorrection.size());
    cv::Mat coordinateAs4Channels = cv::Mat(msg->rows, msg->columns, CV_32FC4);

    auto timeStartInsertingChannels = clock::now();

    cv::insertChannel(cameraFrameCoordinatesX, coordinateAs4Channels, 0);
    cv::insertChannel(cameraFrameCoordinatesY, coordinateAs4Channels, 1);
    cv::insertChannel(depthMapAfterLinearCorrection, coordinateAs4Channels, 2);
    cv::insertChannel(m_ones, coordinateAs4Channels, 3);

    auto timeEndInsertingChannels = clock::now();
    auto durationInsertingChannels = std::chrono::duration_cast<std::chrono::microseconds>(timeEndInsertingChannels - timeStartInsertingChannels).count();
    RCLCPP_INFO(this->get_logger(), "Inserted channels into 4-channel matrix in %ld microseconds", durationInsertingChannels);
    // Alternatively:
    //cv::merge(std::vector<cv::Mat>{cameraFrameCoordinatesX, cameraFrameCoordinatesY, depthMapAfterLinearCorrection, m_ones}  , coordinateAs4Channels);

    CV_Assert(coordinateAs4Channels.isContinuous());
    const auto numberOfPixels {coordinateAs4Channels.rows * coordinateAs4Channels.cols};
    cv::Mat points4xN = coordinateAs4Channels.reshape(1, numberOfPixels).t(); //Now: N rows, 4 columns, 1 channel

    /*
    std::ofstream ofsBefore("pointcloud_before_transformation_" + std::to_string(m_frameCounter) + ".txt");
    if (ofsBefore.is_open())
    {
        for (int i = 0; i < points4xN.cols; ++i)
        {
            ofsBefore << points4xN.at<float>(0,i) << " " << points4xN.at<float>(1,i) << " " << points4xN.at<float>(2,i) << "\n";
        }
        ofsBefore.close();
        RCLCPP_INFO(this->get_logger(), "Saved point cloud before transformation to pointcloud_before_transformation_%d.txt", m_frameCounter);
    }
    */

    cv::Quatd quaternion{msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z};
    cv::Mat transformationMatrix = cv::Mat(quaternion.toRotMat4x4());
    transformationMatrix.convertTo(transformationMatrix, CV_32F);

    //std::cout << "Transformation matrix: \n" << transformationMatrix << std::endl;
    transformationMatrix.at<float>(0,3) = static_cast<float>(msg->pose.pose.position.x);
    transformationMatrix.at<float>(1,3) = static_cast<float>(msg->pose.pose.position.y);
    transformationMatrix.at<float>(2,3) = static_cast<float>(msg->pose.pose.position.z);
    transformationMatrix.convertTo(transformationMatrix, CV_32F);

    //std::cout << "Transformation matrix: \n" << transformationMatrix << std::endl;

    /*
    auto timeStartMatrixMultiplicationEigen = clock::now(); 
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> transformationMarixEigen(transformationMatrix.ptr<float>(), transformationMatrix.rows, transformationMatrix.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> points4xNEigen(points4xN.ptr<float>(), points4xN.rows, points4xN.cols);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points4xNTransformedEigen = transformationMarixEigen * points4xNEigen;
    auto timeEndMatrixMultiplicationEigen = clock::now();
    auto durationMatrixMultiplicationEigen = std::chrono::duration_cast<std::chrono::microseconds>(timeEndMatrixMultiplicationEigen - timeStartMatrixMultiplicationEigen).count();
    RCLCPP_INFO(this->get_logger(), "Matrix multiplication with Eigen took %ld microseconds", durationMatrixMultiplicationEigen);
    */


    auto timeStartMatrixMultiplication= clock::now();
    cv::Mat points4xNTransformed = (transformationMatrix* points4xN).t();
    auto timeEndMatrixMultiplication = clock::now();
    auto durationMatrixMultiplication = std::chrono::duration_cast<std::chrono::microseconds>(timeEndMatrixMultiplication - timeStartMatrixMultiplication).count();
    RCLCPP_INFO(this->get_logger(), "Matrix multiplication took %ld microseconds", durationMatrixMultiplication);



    cv::Mat coordinatesInWorldAs4Channels = points4xNTransformed.reshape(4).reshape(4, msg->rows);

    cv::Rect roi(msg->depth_map_col_min, msg->depth_map_row_min, msg->depth_map_col_max - msg->depth_map_col_min, msg->depth_map_row_max - msg->depth_map_row_min);

    cv::Mat coordinatesInWorldAs4ChannelsCroped = coordinatesInWorldAs4Channels(roi).clone();
    cv::Mat imageSegmentedByObjectIdsCroped = imageSegmentedByObjectIds(roi).clone();

    RCLCPP_INFO(this->get_logger(), "Cropped coordinate matrix is of size: rows=%d, cols=%d", coordinatesInWorldAs4ChannelsCroped.rows, coordinatesInWorldAs4ChannelsCroped.cols);
    RCLCPP_INFO(this->get_logger(), "Image segmented by object IDs size: rows=%d, cols=%d", imageSegmentedByObjectIdsCroped.rows, imageSegmentedByObjectIdsCroped.cols);

    std::vector<std::vector<geometry_msgs::msg::Point32>> pointsPerObject(numberOfObjects);


    for (auto pointCloud: pointsPerObject)
    {
        pointCloud.reserve(500000); //Preallocate space for points
    }

    for (int r = 0; r < imageSegmentedByObjectIdsCroped.rows; ++r)
    {
        const uint16_t* row_ptr = imageSegmentedByObjectIdsCroped.ptr<uint16_t>(r);

        for (int c = 0; c < imageSegmentedByObjectIdsCroped.cols; ++c)
        {
            if (row_ptr[c] == 0)
            {
                continue; //Background class, skip
            }

            const auto objectClassIndex = row_ptr[c] - 1; //Assuming class IDs start from 1

            geometry_msgs::msg::Point32 point;
            const auto& pointInWorld = coordinatesInWorldAs4ChannelsCroped.at<cv::Vec4f>(r,c);
            point.x = pointInWorld[0];
            point.y = pointInWorld[1];
            point.z = pointInWorld[2];
            pointsPerObject[objectClassIndex].emplace_back(std::move(point));
            
        }
    }
    
    auto timeEndTransformation = clock::now();
    auto durationOptimization = std::chrono::duration_cast<std::chrono::microseconds>(timeEndTransformation - timeStartTransformation).count();
    RCLCPP_INFO(this->get_logger(), "Transformed points to world frame in %ld microseconds", durationOptimization);

    /*/
    std::ofstream ofs(pathFilePointCloudForDebug);
    if (ofs.is_open())
    {
        for (auto objectId{0}; objectId < numberOfObjects; ++objectId)
        {         
            const auto classId = msg->list_of_classes[objectId];
            for (const auto& point : pointsPerObject[objectId])
            {
                ofs << point.x << " " << point.y << " " << point.z  << " "<< classId << "\n";
            }
        }
        ofs.close();
        RCLCPP_INFO(this->get_logger(), "Saved point cloud for class 0 to %s", pathFilePointCloudForDebug.c_str());
    }
    */

    //Depth map correciton using regressino results

    //Forming a 3-channel CV32FC3 array with xyz coordinates

    ////reshape this array to a pointcloud layout:
    //CV_Assert(xyz.type() == CV_32FC3);
    //CV_Assert(xyz.isContinuous());  // important!
    //int N = xyz.rows * xyz.cols;
//
    //// Now: N rows, 3 columns, 1 channel
    //cv::Mat points_Nx3 = xyz.reshape(1, N);



    m_isTheFirstFrameBeingProcessed = false;
    m_frameCounter++;
}


