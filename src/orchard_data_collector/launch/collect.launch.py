"""
Launch: BC Data Collector (runs on laptop)

Usage:
    ros2 launch orchard_data_collector collect.launch.py

Then in another terminal:
    ros2 service call /data_collector/toggle_recording std_srvs/srv/SetBool "{data: true}"
    ros2 service call /data_collector/toggle_recording std_srvs/srv/SetBool "{data: false}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument('image_topic',
            default_value='/w200_0100/sensors/camera_0/color/image'),
        DeclareLaunchArgument('odom_topic',
            default_value='/w200_0100/platform/odom/filtered'),
        DeclareLaunchArgument('save_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('skip_stationary', default_value='false'),
        DeclareLaunchArgument('image_dir',
            default_value='/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/images'),
        DeclareLaunchArgument('labels_file',
            default_value='/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/labels.csv'),

        Node(
            package='orchard_data_collector',
            executable='data_collector',
            name='data_collector',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic':        LaunchConfiguration('image_topic'),
                'odom_topic':         LaunchConfiguration('odom_topic'),
                'save_rate_hz':       LaunchConfiguration('save_rate_hz'),
                'auto_start':         LaunchConfiguration('auto_start'),
                'skip_stationary':    LaunchConfiguration('skip_stationary'),
                'image_dir':          LaunchConfiguration('image_dir'),
                'labels_file':        LaunchConfiguration('labels_file'),
                'image_width':        256,
                'image_height':       256,
                'sync_slop_sec':      0.1,
                'sync_queue_size':    10,
                'min_linear_vel':     0.01,
            }],
        ),
    ])
