cmake_minimum_required(VERSION 3.5)
project(orchard_worlds)

find_package(ament_cmake REQUIRED)

# Install world files
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# Install models
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
