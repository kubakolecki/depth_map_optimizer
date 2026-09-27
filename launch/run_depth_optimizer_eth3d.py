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
                        {'do_run_rigorous_optimization': True},
                        {'depth_map_scale_factor': 2},
                        {'min_number_of_map_points': 20},
                        {'opencv_number_of_threads': 16},
                        {'number_of_ceres_iterations': 2},
                        {'number_of_ceres_iterations_second_step': 0},
                        {'regression_outlier_threshold': 0.3},
                        {'regression_outlier_probability': 0.75},
                        {'map_point_difference_threshold': 0.2},
                        {'depth_map_uncertainty_coefficient': 0.01},
                        {'optimization_approach': 'WITHOUT_SCALE_ESTIMATION'}, #WITH_SCALE_ESTIMATION, WITHOUT_SCALE_ESTIMATION 
                        {'ceres_loss_function_depth_map': 'TRIVIAL'}, #TRIVIAL, HUBER, CAUCHY,TUKEY
                        {'ceres_loss_function_depth_map_parameter': 3.0},
                        {'ceres_loss_function_map_points': 'TRIVIAL'}, #TRIVIAL, HUBER, CAUCHY,TUKEY
                        {'ceres_loss_function_map_points_parameter': 3.0},
                        {'ceres_loss_function_map_points_second_step': 'TRIVIAL'}, #TRIVIAL, HUBER, CAUCHY,TUKEY
                        {'ceres_loss_function_map_points_second_step_parameter': 1.0},
                        {'do_save_depth_maps_to_files': True},
                        {'do_save_optimization_reports_to_files': True},
                        {'path_to_depthmap_directory': '/datadisk/data/agh_projects/20260825_depth_maps_datasets_eth3d/variant_tests_improved_optimization/depth_maps/table_3' },
                        {'path_to_optimization_reports_directory': '/datadisk/data/agh_projects/20260825_depth_maps_datasets_eth3d/variant_tests_improved_optimization/depth_maps/table_3' },
                        ],
            emulate_tty=True
        )
    ])
