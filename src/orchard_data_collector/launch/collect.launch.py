"""
Launch: BC Data Collection
===========================
Starts recording immediately. Press Ctrl+C to stop.

Usage:
    # Default (uses Warthog w200_0100 topics)
    ros2 launch orchard_data_collector collect.launch.py

    # Override topics if needed:
    ros2 launch orchard_data_collector collect.launch.py \
        image_topic:=/some/other/image \
        cmd_vel_topic:=/some/other/cmd_vel

    # Override save rate:
    ros2 launch orchard_data_collector collect.launch.py save_rate_hz:=10.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Launch arguments ────────────────────────────────────────────
        DeclareLaunchArgument(
            'image_topic',
            default_value='/w200_0100/sensors/camera_0/color/image',
            description='Camera image topic'),

        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/w200_0100/rc_teleop/cmd_vel',
            description='RC teleop velocity command topic'),

        DeclareLaunchArgument(
            'save_rate_hz',
            default_value='5.0',
            description='Frames per second to save'),

        DeclareLaunchArgument(
            'image_dir',
            default_value='/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/images',
            description='Directory to save images'),

        DeclareLaunchArgument(
            'labels_file',
            default_value='/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/data/raw/labels.csv',
            description='Path to labels CSV'),

        DeclareLaunchArgument(
            'skip_stationary',
            default_value='true',
            description='Skip frames where robot is not moving'),

        # ── Data collector node ─────────────────────────────────────────
        Node(
            package='orchard_data_collector',
            executable='data_collector',
            name='data_collector',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'save_rate_hz': LaunchConfiguration('save_rate_hz'),
                'image_dir': LaunchConfiguration('image_dir'),
                'labels_file': LaunchConfiguration('labels_file'),
                'skip_stationary': LaunchConfiguration('skip_stationary'),
                'image_width': 256,
                'image_height': 256,
                'image_stale_sec': 0.5,
                'cmd_vel_stale_sec': 2.0,
                'min_linear_vel': 0.01,
            }],
        ),
    ])
