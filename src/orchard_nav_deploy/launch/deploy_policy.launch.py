"""
Launch: Deploy BC Policy
=========================
Runs the trained policy on the Warthog with human override capability.
No DAgger logging — just pure deployment.

Usage:
    ros2 launch orchard_nav_deploy deploy_policy.launch.py

    # With custom topics:
    ros2 launch orchard_nav_deploy deploy_policy.launch.py \
        image_topic:=/front_camera/image_raw \
        human_cmd_topic:=/warthog_velocity_controller/cmd_vel_joy
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Arguments ───────────────────────────────────────────────────
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('human_cmd_topic', default_value='/joy_teleop/cmd_vel'),
        DeclareLaunchArgument('checkpoint_path',
            default_value='/home/divyanthlg/projects/orchard_navigation/vae_stabilityai/checkpoints/best.pt'),

        # ── Policy node ─────────────────────────────────────────────────
        Node(
            package='orchard_nav_deploy',
            executable='policy_node',
            name='policy_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'model_project_path': '/home/divyanthlg/projects/orchard_navigation/vae_stabilityai',
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'vae_model_id': 'stabilityai/sd-vae-ft-mse',
                'latent_dim': 128,
                'action_dim': 2,
                'extra_dim': 0,
                'hidden_dim': 256,
                'image_size': 256,
                'max_linear_vel': 1.0,
                'max_angular_vel': 0.5,
                'inference_rate_hz': 10.0,
            }],
        ),

        # ── Cmd vel mux ────────────────────────────────────────────────
        Node(
            package='orchard_nav_deploy',
            executable='cmd_vel_mux',
            name='cmd_vel_mux',
            output='screen',
            parameters=[{
                'policy_cmd_topic': '/policy/cmd_vel',
                'human_cmd_topic': LaunchConfiguration('human_cmd_topic'),
                'output_cmd_topic': '/cmd_vel',
                'human_timeout_sec': 0.5,
                'joy_linear_deadzone': 0.05,
                'joy_angular_deadzone': 0.05,
            }],
        ),

        # ── Status display ──────────────────────────────────────────────
        Node(
            package='orchard_nav_deploy',
            executable='dagger_status_display',
            name='dagger_status_display',
            output='screen',
        ),
    ])
