from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # -------------------------------------------------------
    # Arguments
    # -------------------------------------------------------
    world_arg = LaunchConfiguration('world')
    rviz_arg = LaunchConfiguration('rviz')

    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value='orchard_4rows_20trees',
        description='World to load'
    )

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Launch RViz2 (true/false)'
    )

    # Robot spawn args
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_z = LaunchConfiguration('z')
    spawn_yaw = LaunchConfiguration('yaw')

    declare_x = DeclareLaunchArgument('x', default_value='0.0', description='Robot spawn X position')
    declare_y = DeclareLaunchArgument('y', default_value='0.0', description='Robot spawn Y position')
    declare_z = DeclareLaunchArgument('z', default_value='0.3', description='Robot spawn Z position')
    declare_yaw = DeclareLaunchArgument('yaw', default_value='0.0', description='Robot spawn yaw (rad)')

    # -------------------------------------------------------
    # Paths
    # -------------------------------------------------------
    orchard_world_path = PathJoinSubstitution([
        FindPackageShare('orchard_worlds'),
        'worlds',
        world_arg
    ])

    # -------------------------------------------------------
    # Include Clearpath Gazebo Simulation
    # -------------------------------------------------------
    clearpath_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('clearpath_gz'),
                'launch',
                'simulation.launch.py'
            ])
        ),
        launch_arguments={
            'world': orchard_world_path,
            'x': spawn_x,
            'y': spawn_y,
            'z': spawn_z,
            'yaw': spawn_yaw,
            'rviz': rviz_arg,
        }.items()
    )

    # -------------------------------------------------------
    # Build LaunchDescription
    # -------------------------------------------------------
    ld = LaunchDescription()
    ld.add_action(declare_world_arg)
    ld.add_action(declare_rviz_arg)
    ld.add_action(declare_x)
    ld.add_action(declare_y)
    ld.add_action(declare_z)
    ld.add_action(declare_yaw)
    ld.add_action(clearpath_launch)

    return ld