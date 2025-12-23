---
title: Tooling - Practical ROS 2
sidebar_label: Tooling
sidebar_position: 3
description: Master ROS 2 command-line tools, URDF robot descriptions, RViz visualization, and the colcon build system for practical robot development.
keywords:
  - ros2-cli
  - urdf
  - rviz
  - colcon
  - launch-files
  - debugging
---

# ROS 2 Tooling

Now that you understand ROS 2's architecture, let's get hands-on with the tools you'll use daily for robot development.

## The ROS 2 Command-Line Interface

The `ros2` command is your primary interface for interacting with running ROS systems.

### Essential Commands

#### 1. Node Management

```bash
# List all running nodes
ros2 node list

# Get detailed info about a node
ros2 node info /camera_driver

# Kill a node
ros2 lifecycle set /node_name shutdown
```

**Example output**:
```
$ ros2 node list
/camera_driver
/object_detector
/motion_planner
```

#### 2. Topic Inspection

```bash
# List all topics
ros2 topic list

# Show message type for a topic
ros2 topic info /camera/image_raw

# Monitor topic publishing rate
ros2 topic hz /camera/image_raw

# Echo messages (print to terminal)
ros2 topic echo /camera/image_raw --once

# Publish a test message
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
```

**Example: Debugging camera issues**
```bash
# Is the camera publishing?
ros2 topic hz /camera/image_raw
# Expected: 30 Hz for a 30 FPS camera

# What's the message structure?
ros2 topic echo /camera/image_raw --once | head -20
```

#### 3. Service Management

```bash
# List all services
ros2 service list

# Get service type
ros2 service type /plan_path

# Call a service
ros2 service call /reset_world std_srvs/srv/Empty
ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{name: 'robot'}"
```

#### 4. Parameter Management

```bash
# List parameters for a node
ros2 param list /motion_planner

# Get parameter value
ros2 param get /motion_planner max_speed

# Set parameter value (runtime configuration)
ros2 param set /motion_planner max_speed 2.0

# Dump all parameters to YAML
ros2 param dump /motion_planner > planner_params.yaml
```

#### 5. Action Commands

```bash
# List all actions
ros2 action list

# Send action goal
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {position: {x: 2.0, y: 3.0}}}"

# Send goal with feedback
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {position: {x: 2.0}}}" --feedback
```

#### 6. System Diagnostics

```bash
# Check for common issues
ros2 doctor

# Check if DDS discovery is working
ros2 daemon status

# Monitor computational graph
ros2 run rqt_graph rqt_graph
```

### Power User Tips

**1. Shell auto-completion**
```bash
# Add to ~/.bashrc for tab completion
source /opt/ros/humble/setup.bash
eval "$(register-python-argcomplete3 ros2)"
```

**2. Output formatting**
```bash
# Get topic info as JSON (parseable)
ros2 topic info /camera/image_raw --verbose --json

# Filter echo output with jq
ros2 topic echo /sensors | jq '.temperature'
```

**3. Namespace filtering**
```bash
# List topics only in /robot1 namespace
ros2 topic list | grep /robot1

# Echo topic with namespace remapping
ros2 topic echo /robot1/camera/image --remap /robot1/camera/image:=/camera/image
```

## URDF: Describing Your Robot

**URDF (Unified Robot Description Format)** is an XML format for defining robot geometry, kinematics, and dynamics.

### Basic URDF Structure

A robot consists of **links** (rigid bodies) connected by **joints**:

```xml
<?xml version="1.0"?>
<robot name="humanoid">

  <!-- Base link (torso) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0" ixz="0" iyy="0.4" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.7 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Neck joint connecting head to torso -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

</robot>
```

### Key URDF Concepts

**1. Link elements**:
- `<visual>`: What you see (rendered geometry)
- `<collision>`: Simplified geometry for physics
- `<inertial>`: Mass and inertia tensor for dynamics

**2. Joint types**:
- `fixed`: No movement (e.g., sensors rigidly attached)
- `revolute`: Rotation with limits (e.g., elbow, knee)
- `continuous`: Unlimited rotation (e.g., wheels)
- `prismatic`: Linear motion (e.g., telescoping arm)
- `floating`: 6-DOF free movement (used for mobile base)

**3. Origin and transforms**:
```xml
<origin xyz="0 0 0.3" rpy="0 0 0"/>
<!-- xyz: position offset (x, y, z) in meters -->
<!-- rpy: orientation in roll-pitch-yaw (radians) -->
```

### Xacro: Macros for URDF

Raw URDF is repetitive. **Xacro** adds macros, variables, and includes:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

  <!-- Define reusable properties -->
  <xacro:property name="arm_length" value="0.4"/>
  <xacro:property name="arm_radius" value="0.05"/>

  <!-- Macro for creating arms -->
  <xacro:macro name="arm" params="prefix parent">
    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
      </visual>
    </link>

    <joint name="${prefix}_shoulder" type="revolute">
      <parent link="${parent}"/>
      <child link="${prefix}_upper_arm"/>
      <limit lower="-3.14" upper="3.14" effort="30" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Use macro for both arms -->
  <xacro:arm prefix="left" parent="base_link"/>
  <xacro:arm prefix="right" parent="base_link"/>

</robot>
```

**Convert Xacro to URDF**:
```bash
xacro robot.urdf.xacro > robot.urdf
```

### Validating URDF

```bash
# Check URDF syntax
check_urdf robot.urdf

# Visualize joint tree
urdf_to_graphiz robot.urdf
```

**Expected output**:
```
robot name is: humanoid
---------- Successfully Parsed XML ---------------
root Link: base_link has 2 child(ren)
    child(1):  left_upper_arm
    child(2):  right_upper_arm
```

## RViz: Visualizing Your Robot

**RViz** is ROS 2's 3D visualization tool for sensor data, robot models, and planning results.

### Launching RViz

```bash
# Basic launch
rviz2

# Load with saved configuration
rviz2 -d my_config.rviz
```

### Key Display Types

1. **RobotModel**: Visualize URDF with joint states
2. **TF**: Show coordinate frame tree
3. **Camera**: Display image topics
4. **LaserScan**: Visualize LiDAR data
5. **PointCloud2**: 3D sensor data
6. **Map**: Occupancy grids for navigation
7. **Path**: Planned trajectories
8. **Marker**: Custom 3D shapes for debugging

### Viewing a Robot Model

**Step 1**: Publish robot description
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.publisher = self.create_publisher(String, '/robot_description', 10)

        # Read URDF file
        with open('robot.urdf', 'r') as f:
            urdf_content = f.read()

        # Publish
        msg = String()
        msg.data = urdf_content
        self.publisher.publish(msg)
```

**Step 2**: Configure RViz
1. Add → RobotModel
2. Set "Robot Description" topic to `/robot_description`
3. Add → TF to see coordinate frames

### Example RViz Configuration

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Robot Description: /robot_description
      Visual Enabled: true

    - Class: rviz_default_plugins/TF
      Show Names: true
      Show Axes: true
      Frame Timeout: 15

    - Class: rviz_default_plugins/Camera
      Image Topic: /camera/image_raw
      Transport Hint: raw

  Global Options:
    Fixed Frame: base_link
```

## Colcon: The Build System

**Colcon** (collective construction) is ROS 2's build tool, replacing ROS 1's `catkin_make`.

### Workspace Structure

```
ros2_workspace/
├── src/
│   ├── my_robot_description/
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── urdf/
│   │       └── robot.urdf
│   └── my_robot_control/
│       ├── package.xml
│       ├── setup.py
│       └── my_robot_control/
│           └── controller_node.py
├── build/         # Compilation artifacts
├── install/       # Installed packages
└── log/           # Build logs
```

### Building Packages

```bash
# Build all packages in workspace
cd ~/ros2_workspace
colcon build

# Build specific package
colcon build --packages-select my_robot_control

# Build with symlink install (for Python, faster iteration)
colcon build --symlink-install

# Build with verbose output
colcon build --event-handlers console_direct+
```

### Sourcing the Workspace

After building, source the install space:

```bash
source ~/ros2_workspace/install/setup.bash

# Add to ~/.bashrc for automatic sourcing
echo "source ~/ros2_workspace/install/setup.bash" >> ~/.bashrc
```

### Package Structure (Python)

**Minimal package**:

```
my_package/
├── package.xml          # Package metadata
├── setup.py             # Python setup script
├── setup.cfg            # setuptools configuration
├── resource/my_package  # Resource marker file
└── my_package/
    ├── __init__.py
    └── my_node.py       # Python node
```

**package.xml**:
```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_package</name>
  <version>1.0.0</version>
  <description>My ROS 2 package</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_python</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

**setup.py**:
```python
from setuptools import setup

package_name = 'my_package'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'my_node = my_package.my_node:main',
        ],
    },
)
```

## Launch Files

**Launch files** start multiple nodes with configured parameters and remappings.

### Python Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_urdf}]
        ),

        # Start camera driver
        Node(
            package='usb_cam',
            executable='usb_cam_node',
            name='camera_driver',
            parameters=[{
                'video_device': '/dev/video0',
                'framerate': 30.0,
            }],
            remappings=[
                ('/image_raw', '/camera/image_raw'),
            ]
        ),

        # Start object detector
        Node(
            package='object_detection',
            executable='detector_node',
            parameters=[{
                'confidence_threshold': 0.7,
                'model_path': '/models/yolov5.pt',
            }]
        ),

        # Start RViz with config
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', '/path/to/config.rviz']
        ),
    ])
```

**Running launch files**:
```bash
ros2 launch my_package robot.launch.py
```

## Debugging Tools

### 1. rqt - GUI Plugin Framework

```bash
# Launch plugin manager
rqt

# Or specific plugins:
rqt_graph       # Computational graph
rqt_console     # Log messages
rqt_plot        # Plot numeric topics
rqt_reconfigure # Dynamic parameter tuning
rqt_tf_tree     # TF frame tree
```

### 2. Bag Files - Record and Replay

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /camera/image_raw /imu/data

# Record to specific file
ros2 bag record -o my_dataset /camera/image_raw

# Replay
ros2 bag play my_dataset.bag

# Replay at half speed
ros2 bag play my_dataset.bag --rate 0.5
```

**Use case**: Record sensor data during testing, replay for offline development.

### 3. Logging

```python
class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Log levels: DEBUG, INFO, WARN, ERROR, FATAL
        self.get_logger().debug('Detailed debug info')
        self.get_logger().info('Normal operation')
        self.get_logger().warn('Something unexpected')
        self.get_logger().error('Recoverable error')
        self.get_logger().fatal('Unrecoverable error')
```

**View logs**:
```bash
# Console output
ros2 run my_package my_node

# Set log level
ros2 run my_package my_node --ros-args --log-level DEBUG

# View logs in rqt_console
rqt_console
```

## Docker for ROS 2 Development

**Dockerfile for ROS 2 Humble**:

```dockerfile
FROM osrf/ros:humble-desktop

# Install additional tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-usb-cam \
    ros-humble-navigation2 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy source code
COPY src/ /ros2_ws/src/

# Build workspace
RUN . /opt/ros/humble/setup.sh && colcon build

# Source workspace on container start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
```

**Running**:
```bash
# Build image
docker build -t my_ros2_robot .

# Run with GUI support (Linux)
docker run -it --rm \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  my_ros2_robot
```

## Summary: Essential Workflow

**Typical development cycle**:

1. **Create package**: `ros2 pkg create my_package --build-type ament_python`
2. **Write code**: Implement nodes in `my_package/` directory
3. **Build**: `colcon build --packages-select my_package`
4. **Source**: `source install/setup.bash`
5. **Run**: `ros2 run my_package my_node`
6. **Debug**: Use `ros2 topic echo`, `rqt_graph`, `rqt_console`
7. **Iterate**: Make changes, rebuild with `--symlink-install` for fast iteration

## Next Steps

You now have the practical tools to develop ROS 2 applications! Continue to [Integration](/docs/module-01-ros2/integration) to learn how ROS 2 connects with simulation (Gazebo, Unity) and AI frameworks (Module 3 & 4).

---

**Key Takeaway**: Master the `ros2` CLI, URDF robot descriptions, RViz visualization, and colcon build system for efficient ROS 2 development. These tools are your daily drivers for building autonomous robots.
