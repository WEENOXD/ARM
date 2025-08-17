#!/usr/bin/env python3
"""
3D Robot Arm CAD Viewer with Real-time Pose Control
A 3D simulation of a robot arm controlled by your left arm movements via VPU pose detection.
Controls: Arrow keys + WAED for manual joint manipulation, or automatic pose control
"""

import pygame
import math
import numpy as np
import cv2
import threading
import time
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Import YOLO for pose detection
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not found. Please install: pip install ultralytics")
    YOLO = None

class PoseDetector:
    """Real-time pose detection using YOLO11 Pose"""
    
    def __init__(self):
        self.model = None
        self.cap = None
        self.left_shoulder_angle = 0.0
        self.left_elbow_angle = 0.0
        self.running = False
        self.thread = None
        
        if YOLO:
            try:
                self.model = YOLO('yolo11n-pose.pt')
                print("🤖 YOLO pose detection initialized")
            except Exception as e:
                print(f"❌ YOLO initialization failed: {e}")
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is the vertex)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0
        
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def start_camera(self):
        """Initialize camera for pose detection"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True
    
    def pose_detection_loop(self):
        """Main pose detection loop running in separate thread"""
        if not self.model or not self.cap:
            return
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            try:
                # Run YOLO inference
                results = self.model(frame, verbose=False)
                
                for r in results:
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        keypoints = r.keypoints[0].xy[0].cpu().numpy()
                        
                        # YOLO keypoint indices for left arm
                        left_shoulder_idx = 5
                        left_elbow_idx = 7
                        left_wrist_idx = 9
                        right_shoulder_idx = 6
                        
                        # Check if all required keypoints are detected
                        if (len(keypoints) > 9 and 
                            np.sum(keypoints[left_shoulder_idx]) > 0 and
                            np.sum(keypoints[left_elbow_idx]) > 0 and
                            np.sum(keypoints[left_wrist_idx]) > 0 and
                            np.sum(keypoints[right_shoulder_idx]) > 0):
                            
                            # Calculate left shoulder angle (left shoulder, right shoulder, left elbow)
                            self.left_shoulder_angle = self.calculate_angle(
                                keypoints[left_shoulder_idx],
                                keypoints[right_shoulder_idx], 
                                keypoints[left_elbow_idx]
                            )
                            
                            # Calculate left elbow angle (left shoulder, left elbow, left wrist)
                            self.left_elbow_angle = self.calculate_angle(
                                keypoints[left_shoulder_idx],
                                keypoints[left_elbow_idx],
                                keypoints[left_wrist_idx]
                            )
                            
                            # Map angles to robot arm range
                            # Shoulder: map to -90 to 90 degrees
                            self.left_shoulder_angle = np.clip(self.left_shoulder_angle - 90, -90, 90)
                            # Elbow: map to -150 to 150 degrees  
                            self.left_elbow_angle = np.clip(self.left_elbow_angle - 180, -150, 150)
                            
                            break
                            
            except Exception as e:
                print(f"Pose detection error: {e}")
            
            time.sleep(0.033)  # ~30 FPS
    
    def start(self):
        """Start pose detection in background thread"""
        if not self.start_camera():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.pose_detection_loop, daemon=True)
        self.thread.start()
        print("📹 Pose detection started")
        return True
    
    def stop(self):
        """Stop pose detection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        print("📹 Pose detection stopped")
    
    def get_angles(self):
        """Get current left arm angles"""
        return self.left_shoulder_angle, self.left_elbow_angle

class RobotArm:
    def __init__(self):
        # Joint angles in degrees
        self.base_angle = 0.0
        self.shoulder_angle = 0.0
        self.elbow_angle = 0.0
        
        # Segment lengths (bicep:forearm = 2.9:2.5)
        self.bicep_length = 2.9
        self.forearm_length = 2.5
        self.base_height = 1.0
        
        # Joint limits (in degrees)
        self.joint_limits = {
            'base': (-180, 180),
            'shoulder': (-90, 90),
            'elbow': (-150, 150)
        }
        
        # Control sensitivity
        self.angle_step = 2.0
        
        # Control mode
        self.pose_control = False

    def update_joint(self, joint, delta):
        """Update joint angle without limits"""
        if joint == 'base':
            self.base_angle += delta
        elif joint == 'shoulder':
            self.shoulder_angle += delta
        elif joint == 'elbow':
            self.elbow_angle += delta
    
    def set_pose_angles(self, shoulder_angle, elbow_angle):
        """Set angles directly from pose detection"""
        self.shoulder_angle = shoulder_angle
        self.elbow_angle = elbow_angle

    def get_end_effector_position(self):
        """Calculate end effector position using forward kinematics"""
        # Convert angles to radians
        base_rad = math.radians(self.base_angle)
        shoulder_rad = math.radians(self.shoulder_angle)
        elbow_rad = math.radians(self.elbow_angle)
        
        # Forward kinematics calculations
        # Base position
        base_x = 0
        base_y = 0
        base_z = self.base_height
        
        # Shoulder position (end of bicep)
        shoulder_x = self.bicep_length * math.cos(shoulder_rad) * math.cos(base_rad)
        shoulder_y = self.bicep_length * math.cos(shoulder_rad) * math.sin(base_rad)
        shoulder_z = base_z + self.bicep_length * math.sin(shoulder_rad)
        
        # End effector position (end of forearm)
        total_elbow_angle = shoulder_rad + elbow_rad
        end_x = shoulder_x + self.forearm_length * math.cos(total_elbow_angle) * math.cos(base_rad)
        end_y = shoulder_y + self.forearm_length * math.cos(total_elbow_angle) * math.sin(base_rad)
        end_z = shoulder_z + self.forearm_length * math.sin(total_elbow_angle)
        
        return {
            'base': (base_x, base_y, base_z),
            'shoulder': (shoulder_x, shoulder_y, shoulder_z),
            'end_effector': (end_x, end_y, end_z)
        }

class RobotArmViewer:
    def __init__(self):
        self.robot_arm = RobotArm()
        self.pose_detector = PoseDetector()
        self.camera_distance = 10.0
        self.camera_angle_x = 20.0
        self.camera_angle_y = 45.0
        
    def init_opengl(self, width, height):
        """Initialize OpenGL settings"""
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Robot Arm - Pose Controlled")
        
        # Set clear color to light gray
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width/height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
    def setup_camera(self):
        """Set up camera position and orientation"""
        glLoadIdentity()
        
        # Calculate camera position
        cam_x = self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        cam_y = self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        cam_z = self.camera_distance * math.sin(math.radians(self.camera_angle_x))
        
        gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                  0, 0, 1.5,            # Look at point (center of robot arm)
                  0, 0, 1)              # Up vector (Z is up)

    def draw_cylinder(self, radius, height, slices=16):
        """Draw a cylinder for robot arm segments"""
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        
        # Draw cylinder body
        gluCylinder(quadric, radius, radius, height, slices, 1)
        
        # Draw bottom cap
        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(quadric, 0, radius, slices, 1)
        glPopMatrix()
        
        # Draw top cap
        glPushMatrix()
        glTranslatef(0, 0, height)
        gluDisk(quadric, 0, radius, slices, 1)
        glPopMatrix()
        
        gluDeleteQuadric(quadric)

    def draw_sphere(self, radius, slices=16, stacks=16):
        """Draw a sphere for joints"""
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

    def draw_robot_arm(self):
        """Draw the complete robot arm"""
        positions = self.robot_arm.get_end_effector_position()
        
        # Draw base
        glPushMatrix()
        glColor3f(0.3, 0.3, 0.8)  # Blue base
        self.draw_cylinder(0.3, self.robot_arm.base_height)
        glPopMatrix()
        
        # Draw base joint and all subsequent parts
        glPushMatrix()
        glTranslatef(0, 0, self.robot_arm.base_height)
        glRotatef(self.robot_arm.base_angle, 0, 0, 1)
        glColor3f(0.8, 0.3, 0.3)  # Red joint
        self.draw_sphere(0.2)
        
        # Draw bicep (upper arm)
        glPushMatrix()
        glRotatef(self.robot_arm.shoulder_angle, 0, 1, 0)
        glColor3f(0.3, 0.8, 0.3)  # Green bicep
        self.draw_cylinder(0.15, self.robot_arm.bicep_length)
        
        # Draw elbow joint
        glPushMatrix()
        glTranslatef(0, 0, self.robot_arm.bicep_length)
        glColor3f(0.8, 0.3, 0.3)  # Red joint
        self.draw_sphere(0.15)
        
        # Draw forearm
        glPushMatrix()
        glRotatef(self.robot_arm.elbow_angle, 0, 1, 0)
        glColor3f(0.8, 0.8, 0.3)  # Yellow forearm
        self.draw_cylinder(0.12, self.robot_arm.forearm_length)
        
        # Draw end effector
        glPushMatrix()
        glTranslatef(0, 0, self.robot_arm.forearm_length)
        glColor3f(0.8, 0.3, 0.8)  # Magenta end effector
        self.draw_sphere(0.1)
        glPopMatrix()  # End effector
        
        glPopMatrix()  # Forearm
        glPopMatrix()  # Elbow joint
        glPopMatrix()  # Bicep
        glPopMatrix()  # Base joint

    def draw_coordinate_frame(self):
        """Draw coordinate frame for reference"""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)
        
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2, 0)
        
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2)
        
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_grid(self):
        """Draw a grid on the ground plane"""
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        grid_size = 10
        for i in range(-grid_size, grid_size + 1):
            # Lines parallel to X axis
            glVertex3f(-grid_size, i, 0)
            glVertex3f(grid_size, i, 0)
            # Lines parallel to Y axis
            glVertex3f(i, -grid_size, 0)
            glVertex3f(i, grid_size, 0)
        
        glEnd()
        glEnable(GL_LIGHTING)

    def handle_input(self, keys):
        """Handle keyboard input for joint control"""
        step = self.robot_arm.angle_step
        
        # Toggle pose control with P key
        if keys[pygame.K_p]:
            self.robot_arm.pose_control = not self.robot_arm.pose_control
            print(f"🎮 Pose control: {'ON' if self.robot_arm.pose_control else 'OFF'}")
            time.sleep(0.2)  # Prevent multiple toggles
        
        # Manual control (only when pose control is off)
        if not self.robot_arm.pose_control:
            # Arrow keys: Left/Right for base, Up/Down for shoulder
            if keys[pygame.K_LEFT]:
                self.robot_arm.update_joint('base', -step)
            if keys[pygame.K_RIGHT]:
                self.robot_arm.update_joint('base', step)
            if keys[pygame.K_UP]:
                self.robot_arm.update_joint('shoulder', step)
            if keys[pygame.K_DOWN]:
                self.robot_arm.update_joint('shoulder', -step)
                
            # WAED: W/E for elbow, A/D for fine base control
            if keys[pygame.K_w]:
                self.robot_arm.update_joint('elbow', step)
            if keys[pygame.K_e]:
                self.robot_arm.update_joint('elbow', -step)
            if keys[pygame.K_a]:
                self.robot_arm.update_joint('base', -step/2)
            if keys[pygame.K_d]:
                self.robot_arm.update_joint('base', step/2)

    def display_info(self):
        """Display joint angles and controls"""
        mode = "POSE" if self.robot_arm.pose_control else "MANUAL"
        print(f"\r[{mode}] Base: {self.robot_arm.base_angle:6.1f}° | "
              f"Shoulder: {self.robot_arm.shoulder_angle:6.1f}° | "
              f"Elbow: {self.robot_arm.elbow_angle:6.1f}°", end="")

    def run(self):
        """Main simulation loop"""
        width, height = 1200, 800
        self.init_opengl(width, height)
        
        # Start pose detection
        if not self.pose_detector.start():
            print("⚠️  Pose detection failed, running in manual mode only")
        
        clock = pygame.time.Clock()
        running = True
        
        print("3D Robot Arm - Pose Controlled")
        print("Controls:")
        print("  P: Toggle pose control ON/OFF")
        print("  Arrow Keys: Base rotation (Left/Right), Shoulder (Up/Down) [manual mode]")
        print("  W/E: Elbow up/down [manual mode]")
        print("  A/D: Fine base control [manual mode]")
        print("  ESC: Exit")
        print("  Mouse: Drag to rotate camera view")
        print()
        
        mouse_dragging = False
        last_mouse_pos = (0, 0)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_dragging = True
                        last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        mouse_dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if mouse_dragging:
                        mouse_pos = pygame.mouse.get_pos()
                        dx = mouse_pos[0] - last_mouse_pos[0]
                        dy = mouse_pos[1] - last_mouse_pos[1]
                        
                        self.camera_angle_y += dx * 0.5
                        self.camera_angle_x += dy * 0.5
                        
                        # Limit vertical camera angle
                        self.camera_angle_x = max(-80, min(80, self.camera_angle_x))
                        
                        last_mouse_pos = mouse_pos
            
            # Handle continuous key presses
            keys = pygame.key.get_pressed()
            self.handle_input(keys)
            
            # Update robot arm with pose data if pose control is enabled
            if self.robot_arm.pose_control:
                shoulder_angle, elbow_angle = self.pose_detector.get_angles()
                self.robot_arm.set_pose_angles(shoulder_angle, elbow_angle)
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up camera
            self.setup_camera()
            
            # Draw scene
            self.draw_grid()
            self.draw_coordinate_frame()
            self.draw_robot_arm()
            
            # Update display
            pygame.display.flip()
            self.display_info()
            
            clock.tick(60)  # 60 FPS
        
        print("\nExiting...")
        self.pose_detector.stop()
        pygame.quit()

if __name__ == "__main__":
    try:
        viewer = RobotArmViewer()
        viewer.run()
    except ImportError as e:
        print(f"Required module not found: {e}")
        print("Please install required packages:")
        print("pip install pygame PyOpenGL PyOpenGL_accelerate numpy opencv-python ultralytics")
    except Exception as e:
        print(f"Error: {e}")
