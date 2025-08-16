# 3D Robot Arm CAD Viewer

A real-time 3D simulation of a robot arm with interactive controls and unrestricted joint movement.

## Features

- **3-DOF Robot Arm**: Rotating base, shoulder, and elbow joints
- **Accurate Proportions**: Bicep to forearm ratio of 2.9:2.5
- **Unrestricted Movement**: Full 360° rotation on all joints
- **Interactive Controls**: Keyboard-based joint manipulation
- **3D Visualization**: OpenGL rendering with lighting and camera controls

## Controls

- **Arrow Keys**: 
  - Left/Right: Base rotation
  - Up/Down: Shoulder joint
- **WAED Keys**:
  - W/E: Elbow joint up/down
  - A/D: Fine base control
- **Mouse**: Drag to rotate camera view
- **ESC**: Exit

## Installation & Running

```bash
# Create virtual environment
python3 -m venv robot_arm_env
source robot_arm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the simulator
python3 robot_arm_3d.py
```

## Joint Specifications

- **Base**: Unlimited rotation
- **Shoulder**: Unlimited rotation 
- **Elbow**: Unlimited rotation
- **Bicep Length**: 2.9 units
- **Forearm Length**: 2.5 units

## Demo

The simulator provides real-time visualization of a 3-DOF robot arm with proper kinematic calculations and collision-free joint movement. All joints can rotate freely without angular constraints.
