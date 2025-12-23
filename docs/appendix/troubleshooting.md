---
id: troubleshooting
title: Troubleshooting
sidebar_label: Troubleshooting
sidebar_position: 3
---

# Troubleshooting

Common issues and solutions for problems you may encounter while working through this book.

## General Issues

### Build Errors

**Problem**: `npm run build` fails with errors

**Solutions**:
1. Verify Node.js version is 18 or higher: `node --version`
2. Clear cache and reinstall: `rm -rf node_modules package-lock.json && npm install`
3. Check for syntax errors in MDX files
4. Ensure all frontmatter is valid YAML

### Import Errors

**Problem**: Custom React components not found

**Solutions**:
1. Verify component files exist in `src/components/`
2. Check import paths are correct: `import Component from '@site/src/components/Component'`
3. Ensure component exports use `export default`
4. Restart dev server after adding new components

## Module 1: ROS 2 Issues

### ROS 2 Installation

**Problem**: ROS 2 Humble not found or installation fails

**Solutions**:
1. Verify Ubuntu 22.04 LTS (Jammy) is installed
2. Follow official installation guide: https://docs.ros.org/en/humble/Installation.html
3. Source ROS 2 setup: `source /opt/ros/humble/setup.bash`
4. Add to `.bashrc` for persistence

**Problem**: `colcon build` fails

**Solutions**:
1. Install colcon: `sudo apt install python3-colcon-common-extensions`
2. Check workspace structure: `src/` directory must exist
3. Resolve missing dependencies: `rosdep install --from-paths src -y`
4. Clean build: `rm -rf build install log`

### Node Communication

**Problem**: Nodes not communicating (topics not visible)

**Solutions**:
1. Check both nodes are running: `ros2 node list`
2. Verify topic names match: `ros2 topic list`
3. Inspect message types: `ros2 topic info /topic_name`
4. Check QoS settings are compatible
5. Verify DDS discovery: `ros2 doctor --report`

**Problem**: Messages not received

**Solutions**:
1. Check subscriber is active: `ros2 topic echo /topic_name`
2. Verify publisher is sending: `ros2 topic hz /topic_name`
3. Ensure message types match exactly
4. Check network configuration (for multi-machine setups)

### URDF Issues

**Problem**: Robot model not loading in RViz

**Solutions**:
1. Validate URDF syntax: `check_urdf robot.urdf`
2. Check file paths are absolute or relative to correct directory
3. Verify all mesh files exist
4. Use `xacro` if using macros: `xacro robot.urdf.xacro > robot.urdf`

## Module 2: Simulation Issues

### Gazebo

**Problem**: Gazebo crashes on startup

**Solutions**:
1. Update graphics drivers
2. Try software rendering: `LIBGL_ALWAYS_SOFTWARE=1 gazebo`
3. Check system requirements: GPU with OpenGL 3.3+ support
4. Reduce physics update rate in world file

**Problem**: Robot model invisible or black

**Solutions**:
1. Check mesh file paths in URDF
2. Verify materials/textures exist
3. Enable shadows and ambient light in Gazebo
4. Inspect collision geometry matches visual geometry

**Problem**: Physics behaving unrealistically

**Solutions**:
1. Adjust friction coefficients in URDF
2. Tune physics engine parameters (gravity, step size)
3. Check mass and inertia values are reasonable
4. Verify joint limits and damping settings

### Unity Simulation

**Problem**: Unity Robotics Hub not connecting to ROS

**Solutions**:
1. Verify ROS TCP Connector is running: `ros2 run ros_tcp_endpoint default_server_endpoint`
2. Check IP address and port configuration match
3. Disable firewall or add exception
4. Ensure ROS_DOMAIN_ID matches between Unity and ROS

**Problem**: Robot movement jerky or unstable

**Solutions**:
1. Increase physics timestep in Unity (Edit → Project Settings → Time)
2. Add damping to joints
3. Check ArticulationBody settings
4. Verify joint drives are configured correctly

## Module 3: AI & Isaac Issues

### Isaac Sim

**Problem**: Isaac Sim won't launch

**Solutions**:
1. Verify NVIDIA GPU with RTX support
2. Update NVIDIA drivers to latest version
3. Check Omniverse Launcher is installed
4. Allocate sufficient GPU memory (8GB+ recommended)

**Problem**: ROS 2 bridge not working

**Solutions**:
1. Enable ROS 2 Bridge extension in Isaac Sim
2. Verify ROS 2 Humble is sourced
3. Check ROS_DOMAIN_ID environment variable
4. Inspect bridge configuration in simulation

### Navigation & SLAM

**Problem**: SLAM map not building

**Solutions**:
1. Verify sensor data is published: `ros2 topic echo /scan`
2. Check sensor TF frames are correct
3. Ensure robot is moving (static robot can't build map)
4. Tune SLAM parameters for environment
5. Verify lidar/camera is not obstructed

**Problem**: Navigation fails or robot gets stuck

**Solutions**:
1. Check costmap is building correctly
2. Verify local and global planner parameters
3. Increase planner patience timeout
4. Check for TF transform errors
5. Ensure robot footprint is configured accurately

### Perception

**Problem**: Object detection not working

**Solutions**:
1. Check camera feed: `ros2 topic hz /camera/image_raw`
2. Verify lighting conditions in simulation
3. Ensure model is loaded correctly
4. Check input image resolution matches model expectations
5. Inspect confidence threshold settings

## Module 4: VLA Issues

### LLM Integration

**Problem**: API calls timing out or failing

**Solutions**:
1. Check API key is valid and has credits
2. Verify network connectivity
3. Add error handling and retries
4. Use smaller, faster models for testing
5. Check rate limits aren't exceeded

**Problem**: Poor robot action quality

**Solutions**:
1. Improve prompt engineering (be more specific)
2. Provide more context in prompts
3. Fine-tune model on domain-specific data
4. Add few-shot examples to prompt
5. Validate action outputs before execution

### Voice Control

**Problem**: Speech recognition inaccurate

**Solutions**:
1. Use high-quality microphone
2. Reduce background noise
3. Try different speech recognition models
4. Add wake word detection
5. Implement confirmation prompts for critical actions

**Problem**: Audio input not detected

**Solutions**:
1. Check microphone permissions
2. Verify audio device in system settings
3. Test with: `arecord -l` and `aplay -l`
4. Ensure PulseAudio or ALSA is configured
5. Check ROS 2 audio bridge is running

## System-Level Issues

### Performance

**Problem**: Simulation running slowly

**Solutions**:
1. Reduce visual quality settings
2. Decrease physics update rate
3. Limit number of sensors
4. Close unnecessary applications
5. Use headless mode for training

**Problem**: High CPU/memory usage

**Solutions**:
1. Monitor with `htop` to identify bottleneck
2. Reduce number of active nodes
3. Decrease sensor publishing rates
4. Use efficient data types (e.g., compressed images)
5. Profile code to find performance issues

### Networking

**Problem**: Multi-machine ROS 2 setup not working

**Solutions**:
1. Ensure all machines on same network
2. Set ROS_DOMAIN_ID to same value everywhere
3. Check firewall rules allow DDS traffic
4. Verify multicast is enabled on network
5. Use `ros2 multicast` command to test

## Getting Help

If you can't resolve your issue:

1. **Check the documentation** - Review relevant module chapters
2. **Search existing issues** - Look in ROS Answers, Stack Overflow
3. **Ask the community** - Post on ROS Discourse or Robotics Stack Exchange
4. **Provide details** - Include error messages, ROS version, OS version, steps to reproduce
5. **Share minimal reproducible example** - Isolate the problem to smallest test case

### Useful Debugging Commands

```bash
# ROS 2 diagnostics
ros2 doctor
ros2 wtf

# List all nodes, topics, services
ros2 node list
ros2 topic list
ros2 service list

# Inspect running system
ros2 node info /node_name
ros2 topic info /topic_name
ros2 topic hz /topic_name

# TF debugging
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo source_frame target_frame

# Parameter inspection
ros2 param list
ros2 param get /node_name parameter_name
```

### Log Files

Check log files for detailed error information:

- **ROS 2 logs**: `~/.ros/log/`
- **Gazebo logs**: `~/.gazebo/`
- **Isaac Sim logs**: `~/.nvidia-omniverse/logs/`
- **System logs**: `/var/log/syslog`
