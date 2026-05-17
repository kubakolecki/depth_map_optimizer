import os
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_optimizer',
            executable='depth_optimizer_node',
            name='depth_optimizer_node',
            output='screen',
            parameters=[{'mapping_data_topic_name': 'slam_deep_mapper/mapping_data'},
                        {'opencv_number_of_threads': 16},
                        {'number_of_ceres_iterations': 3},
                        {'regression_outlier_threshold': 0.2},
                        {'regression_outlier_probability': 0.5},
                        {'ceres_loss_function_depth_map': 'CAUCHY'}, #TRIVIAL, HUBER, CAUCHY,TUKEY
                        {'ceres_loss_function_depth_map_parameter': 2.0},
                        {'ceres_loss_function_map_points': 'CAUCHY'}, #TRIVIAL, HUBER, CAUCHY,TUKEY
                        {'ceres_loss_function_map_points_parameter': 2.0},
                        {'do_save_depth_maps_to_files': True},
                        {'do_save_optimization_reports_to_files': True},
                        {'path_to_depthmap_directory': '/datadisk/data/agh_projects/20260407_depth_map_datasets/depth_maps/2026_04_07-14_05_46' },
                        {'path_to_optimization_reports_directory': '/datadisk/data/agh_projects/20260407_depth_map_datasets/depth_maps/2026_04_07-14_05_46' }
                        ],
            emulate_tty=True
        )
    ])