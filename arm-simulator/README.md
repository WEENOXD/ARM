# 3D Robot Arm CAD Viewer

A real-time 3D simulation of a robot arm with interactive controls.

## Features

- **3-DOF Robot Arm**: Rotating base, shoulder, and elbow joints
- **Accurate Proportions**: Bicep to forearm ratio of 2.9:2.5
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

- **Base**: 360° rotation (-180° to +180°)
- **Shoulder**: 180° range (-90° to +90°) 
- **Elbow**: 300° range (-150° to +150°)
- **Bicep Length**: 2.9 units
- **Forearm Length**: 2.5 units