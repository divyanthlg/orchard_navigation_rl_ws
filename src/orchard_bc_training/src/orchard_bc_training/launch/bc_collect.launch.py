"""
Launch: v0.7 BC Data Collector (compressed image input)

    ros2 launch orchard_bc_training bc_collect.launch.py

Toggle:
    ros2 service call /bc_data_collector/toggle_recording \
        std_srvs/srv/SetBool "{data: true}"
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_DATA_DIR = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/data/raw')


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('HF_HUB_OFFLINE', '1'),
        SetEnvironmentVariable('TRANSFORMERS_OFFLINE', '1'),

        DeclareLaunchArgument('image_topic',
            default_value='/sensors/camera_0/color/compressed'),
        DeclareLaunchArgument('odom_topic',
            default_value='/platform/odom/filtered'),
        DeclareLaunchArgument('save_rate_hz', default_value='2.0'),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('skip_stationary', default_value='false'),
        DeclareLaunchArgument('image_dir',
            default_value=os.path.join(DEFAULT_DATA_DIR, 'images')),
        DeclareLaunchArgument('labels_file',
            default_value=os.path.join(DEFAULT_DATA_DIR, 'labels.csv')),

        Node(
            package='orchard_bc_training',
            executable='bc_data_collector',
            name='bc_data_collector',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic':     LaunchConfiguration('image_topic'),
                'odom_topic':      LaunchConfiguration('odom_topic'),
                'save_rate_hz':    LaunchConfiguration('save_rate_hz'),
                'auto_start':      LaunchConfiguration('auto_start'),
                'skip_stationary': LaunchConfiguration('skip_stationary'),
                'image_dir':       LaunchConfiguration('image_dir'),
                'labels_file':     LaunchConfiguration('labels_file'),
                'image_width':     256,
                'image_height':    256,
                'sync_slop_sec':   0.1,
                'sync_queue_size': 10,
                'min_linear_vel':  0.01,
            }],
        ),
    ])
