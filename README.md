# Orchard Navigation ROS 2 Workspace

ROS 2 packages for behaviour cloning data collection and policy deployment
on the Clearpath Warthog in orchard environments.

## Packages

- `orchard_data_collector` — BC data collection (camera + teleop → labelled dataset)
- `orchard_nav_deploy` — Trained policy deployment + DAgger data aggregation

## Build
```bash
cd ~/ros2/orchard_navigation_rl_ws
colcon build
source install/setup.bash
```
