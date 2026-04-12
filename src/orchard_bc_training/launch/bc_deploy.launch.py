"""
Launch: v0.7 BC Policy Deploy (offline-safe, with visualization)

    ros2 launch orchard_bc_training bc_deploy.launch.py

View the overlay:
    ros2 run rqt_image_view rqt_image_view /bc_viz/image
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_CHECKPOINT = os.path.expanduser(
    '~/ros2/orchard_navigation_rl_ws/checkpoints/best.pt')


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('HF_HUB_OFFLINE', '1'),
        SetEnvironmentVariable('TRANSFORMERS_OFFLINE', '1'),

        DeclareLaunchArgument('image_topic',
            default_value='/w200_0100/sensors/camera_0/color/image'),
        DeclareLaunchArgument('odom_topic',
            default_value='/w200_0100/platform/odom/filtered'),
        DeclareLaunchArgument('human_cmd_topic',
            default_value='/w200_0100/rc_teleop/cmd_vel'),
        DeclareLaunchArgument('output_cmd_topic',
            default_value='/w200_0100/cmd_vel'),
        DeclareLaunchArgument('checkpoint_path',
            default_value=DEFAULT_CHECKPOINT),
        DeclareLaunchArgument('vae_model_id', default_value='',
            description='Leave empty to auto-resolve: workspace models/ if '
                        'present, else stabilityai/sd-vae-ft-mse from HF cache.'),

        Node(
            package='orchard_bc_training',
            executable='bc_policy_node',
            name='bc_policy_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic':        LaunchConfiguration('image_topic'),
                'odom_topic':         LaunchConfiguration('odom_topic'),
                'checkpoint_path':    LaunchConfiguration('checkpoint_path'),
                'vae_model_id':       LaunchConfiguration('vae_model_id'),
                'seq_len':            13,
                'image_size':         256,
                'max_linear_vel':     1.0,
                'max_angular_vel':    0.5,
                'perception_rate_hz': 2.0,
                'command_rate_hz':    10.0,
                'output_cmd_topic':   '/bc_policy/cmd_vel',
            }],
        ),

        Node(
            package='orchard_bc_training',
            executable='bc_cmd_vel_mux',
            name='bc_cmd_vel_mux',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'policy_cmd_topic':     '/bc_policy/cmd_vel',
                'human_cmd_topic':      LaunchConfiguration('human_cmd_topic'),
                'output_cmd_topic':     LaunchConfiguration('output_cmd_topic'),
                'human_timeout_sec':    0.5,
                'joy_linear_deadzone':  0.05,
                'joy_angular_deadzone': 0.05,
            }],
        ),

        Node(
            package='orchard_bc_training',
            executable='bc_status_display',
            name='bc_status_display',
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='orchard_bc_training',
            executable='bc_viz_node',
            name='bc_viz',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic':      LaunchConfiguration('image_topic'),
                'odom_topic':       LaunchConfiguration('odom_topic'),
                'policy_cmd_topic': '/bc_policy/cmd_vel',
                'output_cmd_topic': LaunchConfiguration('output_cmd_topic'),
                'viz_topic':        '/bc_viz/image',
                'max_linear_vel':   1.0,
                'max_angular_vel':  0.5,
            }],
        ),
    ])
