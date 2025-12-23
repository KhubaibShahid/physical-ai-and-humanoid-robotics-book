---
title: Tooling - Practical Simulation
sidebar_label: Tooling
sidebar_position: 3
description: Hands-on with Gazebo and Unity - create worlds, spawn robots, configure sensors, and build realistic test environments for your humanoid robot.
keywords:
  - gazebo-tutorial
  - unity-tutorial
  - world-building
  - sensor-configuration
---

# Simulation Tooling

Let's get hands-on with Gazebo and Unity. This chapter provides practical workflows for building simulation environments.

## Gazebo Essentials

### Installation

```bash
# Install Gazebo Fortress with ROS 2 Humble
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs gazebo11

# Verify installation
gazebo --version
```

### Basic Workflow

**1. Launch Gazebo**:

```bash
# Empty world
gazebo

# With specific world file
gazebo worlds/cafe.world

# Headless (no GUI, faster)
gzserver worlds/empty.world
```

**2. World Files** - Define environment:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="humanoid_test_world">
    <!-- Physics settings -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <direction>0 0 -1</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add obstacles -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
          </material>
        </visual>
      </link>
      <static>true</static>
    </model>
  </world>
</sdf>
```

**3. Model SDF Files** - Robot description:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="simple_humanoid">
    <!-- Base link (torso) -->
    <link name="torso">
      <pose>0 0 1.0 0 0 0</pose>
      <inertial>
        <mass>50.0</mass>
        <inertia>
          <ixx>1.0</ixx><iyy>1.0</iyy><izz>0.5</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box><size>0.3 0.2 0.6</size></box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box><size>0.3 0.2 0.6</size></box>
        </geometry>
      </visual>

      <!-- Camera sensor -->
      <sensor name="camera" type="camera">
        <update_rate>30</update_rate>
        <camera>
          <horizontal_fov>1.57</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
          </image>
        </camera>
        <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
          <ros>
            <namespace>/robot</namespace>
            <remapping>image_raw:=camera/image</remapping>
          </ros>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

**4. Launch with ROS 2**:

```python
# robot_sim.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start Gazebo
        IncludeLaunchDescription(
            'launch/gazebo.launch.py',
            launch_arguments={'world': 'humanoid_test_world.sdf'}.items()
        ),

        # Spawn robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robot', '-file', 'robot.sdf',
                      '-x', '0', '-y', '0', '-z', '1.0']
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'use_sim_time': True}]
        ),
    ])
```

### Gazebo Plugins

**Common plugins**:

**Differential drive**:
```xml
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <ros>
    <namespace>/robot</namespace>
  </ros>
  <left_joint>left_wheel_joint</left_joint>
  <right_joint>right_wheel_joint</right_joint>
  <wheel_separation>0.4</wheel_separation>
  <wheel_diameter>0.2</wheel_diameter>
  <max_wheel_torque>20</max_wheel_torque>
  <publish_odom>true</publish_odom>
  <publish_odom_tf>true</publish_odom_tf>
  <odometry_frame>odom</odometry_frame>
  <robot_base_frame>base_link</robot_base_frame>
</plugin>
```

**Joint state publisher**:
```xml
<plugin name="joint_states" filename="libgazebo_ros_joint_state_publisher.so">
  <ros>
    <namespace>/robot</namespace>
    <remapping>joint_states:=joint_states</remapping>
  </ros>
  <update_rate>100</update_rate>
</plugin>
```

## Unity Essentials

### Installation

1. **Download Unity Hub**: [unity.com/download](https://unity.com/download)
2. **Install Unity 2022.3 LTS** via Hub
3. **Create new project**: 3D (HDRP)

**Add Unity Robotics Hub**:

```
1. Window → Package Manager
2. Add package from git URL:
   https://github.com/Unity-Technologies/ROS-TCP-Connector.git
3. Import sample assets (optional)
```

### Basic Workflow

**1. Create scene**:

```
File → New Scene → HDRP
GameObject → 3D Object → Plane (ground)
GameObject → Light → Directional Light
```

**2. Import robot URDF**:

```
Packages → URDF Importer → Import URDF
Select robot.urdf → Import
```

**3. Configure ArticulationBody** (robot physics):

```csharp
public class RobotController : MonoBehaviour {
    ArticulationBody[] joints;

    void Start() {
        joints = GetComponentsInChildren<ArticulationBody>();

        foreach (var joint in joints) {
            joint.solverIterations = 10;
            joint.solverVelocityIterations = 10;
        }
    }

    void FixedUpdate() {
        // Control joint positions
        var drive = joints[0].xDrive;
        drive.target = Mathf.Sin(Time.time) * 45f;
        joints[0].xDrive = drive;
    }
}
```

**4. Setup ROS 2 connection**:

```csharp
using Unity.Robotics.ROSTCPConnector;

public class ROSConnection : MonoBehaviour {
    void Start() {
        ROSConnection.GetOrCreateInstance().Connect();
    }
}
```

**ROS 2 side**:
```bash
# Start ROS TCP endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1
```

### Publishing Sensor Data

**Camera publisher**:

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraPublisher : MonoBehaviour {
    ROSConnection ros;
    Camera cam;
    float publishRate = 30f;
    float timer = 0f;

    void Start() {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>("camera/image_raw");
        cam = GetComponent<Camera>();
    }

    void Update() {
        timer += Time.deltaTime;
        if (timer >= 1f / publishRate) {
            PublishImage();
            timer = 0f;
        }
    }

    void PublishImage() {
        RenderTexture rt = new RenderTexture(640, 480, 24);
        cam.targetTexture = rt;
        cam.Render();

        Texture2D image = new Texture2D(640, 480, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        image.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
        image.Apply();

        ImageMsg msg = new ImageMsg {
            header = new HeaderMsg { stamp = GetROSTime() },
            height = 480,
            width = 640,
            encoding = "rgb8",
            data = image.GetRawTextureData()
        };

        ros.Publish("camera/image_raw", msg);

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
    }
}
```

### Domain Randomization

**Randomize lighting**:

```csharp
using UnityEngine.Perception.Randomization.Randomizers;

public class LightRandomizer : Randomizer {
    public FloatParameter intensity = new FloatParameter { value = new UniformSampler(0.5f, 1.5f) };
    public ColorRgbParameter color = new ColorRgbParameter();

    Light directionalLight;

    protected override void OnIterationStart() {
        if (directionalLight == null)
            directionalLight = GameObject.Find("Directional Light").GetComponent<Light>();

        directionalLight.intensity = intensity.Sample();
        directionalLight.color = color.Sample();
    }
}
```

**Randomize textures**:

```csharp
public class TextureRandomizer : Randomizer {
    public Texture2D[] textures;

    protected override void OnIterationStart() {
        Renderer[] renderers = FindObjectsOfType<Renderer>();

        foreach (var renderer in renderers) {
            if (renderer.CompareTag("Randomize")) {
                Texture2D randomTexture = textures[Random.Range(0, textures.Length)];
                renderer.material.mainTexture = randomTexture;
            }
        }
    }
}
```

## Building Test Environments

### Gazebo: Obstacle Course

```xml
<!-- obstacle_course.world -->
<world name="obstacle_course">
  <!-- Stairs -->
  <model name="stairs">
    <pose>5 0 0 0 0 0</pose>
    <static>true</static>
    <!-- Generate steps programmatically or use mesh -->
    <link name="step1">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>1 2 0.2</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>1 2 0.2</size></box></geometry>
      </visual>
    </link>
    <!-- More steps... -->
  </model>

  <!-- Narrow passage -->
  <model name="narrow_passage">
    <pose>10 0 0 0 0 0</pose>
    <static>true</static>
    <!-- Two walls -->
  </model>

  <!-- Ramp -->
  <model name="ramp">
    <pose>15 0 0 0 0.2 0</pose>  <!-- 0.2 rad = ~11 degrees -->
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>3 2 0.1</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>3 2 0.1</size></box></geometry>
      </visual>
    </link>
  </model>
</world>
```

### Unity: Indoor Scene

```csharp
public class ProceduralRoom : MonoBehaviour {
    public GameObject wallPrefab;
    public GameObject furniturePrefab;

    void Start() {
        GenerateRoom(10f, 10f, 3f);  // 10x10m, 3m height
    }

    void GenerateRoom(float width, float depth, float height) {
        // Create walls
        CreateWall(new Vector3(0, height/2, -depth/2), new Vector3(width, height, 0.1f));
        CreateWall(new Vector3(0, height/2, depth/2), new Vector3(width, height, 0.1f));
        CreateWall(new Vector3(-width/2, height/2, 0), new Vector3(0.1f, height, depth));
        CreateWall(new Vector3(width/2, height/2, 0), new Vector3(0.1f, height, depth));

        // Add random furniture
        for (int i = 0; i < 5; i++) {
            Vector3 pos = new Vector3(
                Random.Range(-width/2 + 1, width/2 - 1),
                0,
                Random.Range(-depth/2 + 1, depth/2 - 1)
            );
            Instantiate(furniturePrefab, pos, Quaternion.identity);
        }
    }

    void CreateWall(Vector3 position, Vector3 scale) {
        GameObject wall = Instantiate(wallPrefab, position, Quaternion.identity);
        wall.transform.localScale = scale;
    }
}
```

## Performance Tips

**Gazebo**:
- Use simple collision shapes (boxes, cylinders) instead of meshes
- Limit sensor update rates (camera at 10 Hz for testing)
- Reduce physics iterations for non-critical objects
- Run headless (gzserver) for training

**Unity**:
- Use LOD groups for distant objects
- Disable unnecessary cameras
- Reduce shadow quality for training
- Use object pooling for frequently spawned objects

## Next Steps

Ready to integrate simulation with ROS 2 navigation and AI? Continue to [Integration](/docs/module-02-digital-twin/integration) to connect your digital twin to real robot workflows.

---

**Key Takeaway**: Master Gazebo's SDF world building and Unity's scene creation to build realistic test environments for your humanoid robot before hardware deployment.
