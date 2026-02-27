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
            parameters=[{'object_data_topic_name': '/slam_deep_mapper/object_data'},
                        {'opencv_number_of_threads': 16}],
            emulate_tty=True
        )
    ])