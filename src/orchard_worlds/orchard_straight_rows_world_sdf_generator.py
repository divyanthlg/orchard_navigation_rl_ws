import random

# --------------------
# USER CONFIG
# --------------------
ROW_SPACING = 2.0
TREE_SPACING = 0.75
START_X = 0
START_Y = 0
Z = -0.1

NUM_ROWS = 4
TREES_PER_ROW = 20
NUM_TREE_MODELS = 16

MESH_DIR = "file:///home/divyanthlg/ros2/orchard_navigation_rl_ws/src/orchard_worlds/models/orchard_model"

OUTPUT_FILE = "worlds/orchard_4rows_20trees.sdf"


# --------------------
# TREE MODEL GENERATOR
# --------------------
def generate_tree_block(model_name, x, y, z, yaw, mesh_id):
    return f"""
    <model name="{model_name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>{MESH_DIR}/tree{mesh_id}_collision.stl</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>{MESH_DIR}/tree{mesh_id}.stl</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>0.3 0.2 0.1 1</ambient>
            <diffuse>0.3 0.2 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    """


# --------------------
# BASE SDF HEADER
# --------------------
sdf_header = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="orchard_dormant_world">
    <physics name="default" type="bullet">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <plugin filename="libignition-gazebo-physics-system.so" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="libignition-gazebo-user-commands-system.so" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="libignition-gazebo-scene-broadcaster-system.so" name="ignition::gazebo::systems::SceneBroadcaster"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>500 500</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>500 500</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.15 1</ambient>
            <diffuse>0.2 0.2 0.15 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Orchard terrain mesh -->
    <model name="orchard_terrain">
      <static>true</static>
      <pose>-10 -10 0 0 0 0</pose>
      <link name="terrain_link">
        <collision name="terrain_collision">
          <geometry>
            <mesh>
              <uri>file:///home/divyanthlg/ros2/orchard_navigation_rl_ws/src/orchard_worlds/models/orchard_model/orchard_world.dae</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </collision>
        <visual name="terrain_visual">
          <geometry>
            <mesh>
              <uri>file:///home/divyanthlg/ros2/orchard_navigation_rl_ws/src/orchard_worlds/models/orchard_model/orchard_world.dae</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
"""


# --------------------
# GENERATE ALL TREES
# --------------------
tree_blocks = ""

for row in range(NUM_ROWS):
    for idx in range(TREES_PER_ROW):

        x = START_X + idx * TREE_SPACING
        y = START_Y + row * ROW_SPACING

        mesh_id = random.randint(1, NUM_TREE_MODELS)
        yaw = random.uniform(0, 2 * 3.14159)

        model_name = f"tree_r{row+1}_t{idx+1}"

        tree_blocks += generate_tree_block(model_name, x, y, Z, yaw, mesh_id)


# --------------------
# FINAL SDF FOOTER
# --------------------
sdf_footer = """
  </world>
</sdf>
"""


# --------------------
# WRITE TO FILE
# --------------------
with open(OUTPUT_FILE, "w") as f:
    f.write(sdf_header)
    f.write(tree_blocks)
    f.write(sdf_footer)

print(f"✔️ SDF world generated: {OUTPUT_FILE}")