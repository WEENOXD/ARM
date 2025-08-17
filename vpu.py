#!/usr/bin/env python3
"""
VPU - Vision Processing Unit
Enhanced pixelated face protection + Full body pose tracking with joint angles
"""

import sys
import subprocess
import importlib.util
import os

def check_and_install_dependencies():
    """Check for required dependencies and install them if missing"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'ultralytics': 'ultralytics'
    }
    
    missing_packages = []
    
    # Check which packages are missing
    for module_name, package_name in required_packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("🔧 Missing dependencies detected:", ', '.join(missing_packages))
        print("🚀 Attempting automatic installation...")
        
        try:
            # Try pip install
            for package in missing_packages:
                print(f"📦 Installing {package}...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    # Try with --user flag
                    print(f"⚠️  Standard install failed, trying with --user flag...")
                    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', package], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"❌ Failed to install {package}")
                        print("💡 Please install manually with:")
                        print(f"   pip install {package}")
                        print("   or")
                        print(f"   pip3 install {package}")
                        return False
                    else:
                        print(f"✅ Successfully installed {package} (user)")
                else:
                    print(f"✅ Successfully installed {package}")
            
            print("🎉 All dependencies installed successfully!")
            print("🔄 Please restart the script to continue...")
            return False  # Require restart after installation
            
        except Exception as e:
            print(f"❌ Error during installation: {e}")
            print("💡 Please install dependencies manually:")
            for package in missing_packages:
                print(f"   pip install {package}")
            return False
    
    return True

# Check dependencies before importing
if not check_and_install_dependencies():
    print("🛑 Please install missing dependencies and run the script again.")
    sys.exit(1)

# Import dependencies after checking
try:
    import cv2
    import numpy as np
    import math
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Some dependencies may still be missing. Please restart the script.")
    sys.exit(1)

def pixelate_region(image, x1, y1, x2, y2, pixel_size=20):
    """Apply heavy pixelation effect to image region"""
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image
    
    # Get region dimensions
    h, w = region.shape[:2]
    
    # Resize down to create large pixels
    small = cv2.resize(region, (max(1, w // pixel_size), max(1, h // pixel_size)), 
                      interpolation=cv2.INTER_LINEAR)
    
    # Resize back up to create pixelated effect
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Apply the pixelated region back to the image
    image[y1:y2, x1:x2] = pixelated
    return image

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate angle using dot product
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms == 0:
        return 0
    
    cos_angle = dot_product / norms
    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range for arccos
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

class PoseEstimator:
    """Real-time pose estimation using YOLO11 Pose with advanced smoothing"""
    
    def __init__(self):
        """Initialize YOLO pose model with smoothing"""
        try:
            # Load YOLO11 pose model (will download automatically)
            self.model = YOLO('yolo11n-pose.pt')  # Use nano model for speed
            
            # Smoothing parameters - optimized for better performance
            self.smoothing_factor = 0.6  # Balanced smoothing
            self.previous_keypoints = None
            self.keypoint_history = []
            self.history_size = 4  # Smaller buffer for responsiveness
            self.confidence_threshold = 0.5
            
            # YOLO pose keypoint indices (COCO format)
            self.keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip", 
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            # Key joint angles to track (YOLO keypoint indices)
            self.angle_joints = {
                "Right Elbow": ([6, 8, 10], (255, 0, 0)),    # Right shoulder, elbow, wrist
                "Left Elbow": ([5, 7, 9], (0, 255, 0)),      # Left shoulder, elbow, wrist  
                "Right Knee": ([12, 14, 16], (0, 0, 255)),   # Right hip, knee, ankle
                "Left Knee": ([11, 13, 15], (255, 255, 0)),  # Left hip, knee, ankle
                "Right Shoulder": ([5, 6, 8], (255, 0, 255)), # Left shoulder, right shoulder, right elbow
                "Left Shoulder": ([6, 5, 7], (0, 255, 255)),  # Right shoulder, left shoulder, left elbow
                "Right Hip": ([6, 12, 14], (255, 128, 0)),   # Right shoulder, hip, knee
                "Left Hip": ([5, 11, 13], (128, 255, 0)),    # Left shoulder, hip, knee
            }
            
            print("🤖 YOLO11 pose detection initialized")
            print("📐 Real-time joint angle calculation ready")
            print("✨ Advanced smoothing algorithms enabled")
            print(f"🎯 Smoothing factor: {self.smoothing_factor}")
            print(f"📊 History buffer: {self.history_size} frames")
            
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            self.model = None
    
    def smooth_keypoints(self, raw_keypoints, confidences=None):
        """Apply advanced smoothing to keypoints"""
        if raw_keypoints is None or len(raw_keypoints) == 0:
            return raw_keypoints
            
        # Initialize if first frame
        if self.previous_keypoints is None:
            self.previous_keypoints = raw_keypoints.copy()
            return raw_keypoints
        
        smoothed_keypoints = raw_keypoints.copy()
        
        # Apply per-keypoint smoothing
        for i in range(len(raw_keypoints)):
            current_kp = raw_keypoints[i]
            previous_kp = self.previous_keypoints[i]
            
            # Skip if current keypoint is invalid
            if np.sum(current_kp) == 0:
                smoothed_keypoints[i] = previous_kp
                continue
                
            # Calculate distance moved (velocity check)
            distance = np.linalg.norm(current_kp - previous_kp)
            
            # Adaptive smoothing based on movement
            if distance > 50:  # Large movement - possible error
                adaptive_smoothing = min(0.9, self.smoothing_factor + 0.2)
            elif distance < 5:  # Small movement - less smoothing needed
                adaptive_smoothing = max(0.3, self.smoothing_factor - 0.2)
            else:  # Normal movement
                adaptive_smoothing = self.smoothing_factor
            
            # Apply exponential smoothing
            smoothed_keypoints[i] = (adaptive_smoothing * previous_kp + 
                                   (1 - adaptive_smoothing) * current_kp)
        
        # Store for next frame
        self.previous_keypoints = smoothed_keypoints.copy()
        
        # Moving average smoothing
        self.keypoint_history.append(smoothed_keypoints.copy())
        if len(self.keypoint_history) > self.history_size:
            self.keypoint_history.pop(0)
        
        # Apply moving average
        if len(self.keypoint_history) > 2:
            avg_keypoints = np.mean(self.keypoint_history, axis=0)
            # Blend with current smoothed result
            smoothed_keypoints = (0.7 * smoothed_keypoints + 0.3 * avg_keypoints)
        
        return smoothed_keypoints
    
    def detect_pose_and_angles(self, frame):
        """Detect pose and calculate all joint angles with smoothing"""
        if not self.model:
            return frame
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            # Process results
            for r in results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    # Get raw keypoints for the first person detected
                    raw_keypoints = r.keypoints[0].xy[0].cpu().numpy()  # Shape: (17, 2)
                    confidences = r.keypoints[0].conf[0].cpu().numpy() if hasattr(r.keypoints[0], 'conf') else None
                    
                    # Apply smoothing
                    smoothed_keypoints = self.smooth_keypoints(raw_keypoints, confidences)
                    
                    # Draw skeleton with smoothed keypoints
                    self.draw_skeleton(frame, smoothed_keypoints)
                    
                    # Calculate and display joint angles
                    y_offset = 100
                    
                    for joint_name, (joint_indices, color) in self.angle_joints.items():
                        try:
                            # Check if all required keypoints are detected
                            if all(i < len(smoothed_keypoints) for i in joint_indices):
                                p1 = smoothed_keypoints[joint_indices[0]]
                                p2 = smoothed_keypoints[joint_indices[1]]  # Vertex
                                p3 = smoothed_keypoints[joint_indices[2]]
                                
                                # Check if keypoints are valid (not zero)
                                if all(np.sum(p) > 0 for p in [p1, p2, p3]):
                                    # Calculate angle
                                    angle = calculate_angle(p1, p2, p3)
                                    
                                    # Display angle on left side
                                    cv2.putText(frame, f"{joint_name}: {int(angle)}°", 
                                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.5, color, 2)
                                    y_offset += 22
                                    
                                    # Draw angle at joint position  
                                    if np.sum(p2) > 0:  # Valid keypoint
                                        cv2.putText(frame, f"{int(angle)}°", 
                                                   (int(p2[0]) + 10, int(p2[1]) - 10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                                   0.4, color, 1)
                                
                        except (IndexError, ValueError, TypeError):
                            continue
                            
        except Exception as e:
            print(f"Pose detection error: {e}")
            
        return frame
    
    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections"""
        # COCO pose connections
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),                # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),    # Legs
            (0, 1), (0, 2), (1, 3), (2, 4)             # Head
        ]
        
        for connection in connections:
            try:
                start_idx, end_idx = connection
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    np.sum(keypoints[start_idx]) > 0 and np.sum(keypoints[end_idx]) > 0):
                    
                    start_point = tuple(map(int, keypoints[start_idx]))
                    end_point = tuple(map(int, keypoints[end_idx]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            except:
                continue
        
        # Draw keypoints with better visualization
        for i, kp in enumerate(keypoints):
            if np.sum(kp) > 0:  # Valid keypoint
                # Draw larger, more visible keypoints
                cv2.circle(frame, tuple(map(int, kp)), 6, (0, 255, 255), -1)
                cv2.circle(frame, tuple(map(int, kp)), 8, (0, 0, 0), 2)

def find_available_cameras():
    """Find all available camera devices that can actually capture frames"""
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices (reduces noise)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                available_cameras.append(i)
                print(f"📹 Found working camera {i} - Resolution: {frame.shape}")
            elif cap.isOpened():
                print(f"⚠️  Camera {i} opens but provides no frames")
            cap.release()
    return available_cameras

def initialize_camera(camera_index):
    """Initialize camera with optimal settings"""
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    return None

def main():
    """VPU - Automatic pixelated face protection + Full body pose tracking"""
    print("🎭 VPU - Vision Processing Unit")
    print("🔒 Enhanced pixelated face protection enabled")
    print("⏱️  5-frame persistence active") 
    print("🤖 Full body pose tracking with joint angles")
    print("📐 Real-time biomechanical analysis")
    print("🎮 Camera controls: ← → arrows to switch cameras")
    
    # Find available cameras
    available_cameras = find_available_cameras()
    if not available_cameras:
        print("❌ No working cameras found!")
        print("💡 Troubleshooting tips:")
        print("   • Check if camera permissions are granted")
        print("   • Close other applications using the camera")
        print("   • Try different camera indices manually")
        return
    
    print(f"🎥 Found {len(available_cameras)} working camera(s): {available_cameras}")
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator()
    
    # Face tracking for persistence
    face_memory = []  # List of (x1, y1, x2, y2, frames_remaining)
    PERSIST_FRAMES = 5
    
    # Camera management
    current_camera_idx = 0
    current_camera_id = available_cameras[current_camera_idx]
    cap = initialize_camera(current_camera_id)
    
    if not cap:
        print(f"❌ Could not initialize camera {current_camera_id}")
        return
    
    # Test if we can actually read from the camera
    test_ret, test_frame = cap.read()
    if not test_ret:
        print(f"❌ Camera {current_camera_id} initialized but cannot capture frames")
        print("💡 Try closing other applications that might be using the camera")
        cap.release()
        return
    
    print(f"✅ Camera {current_camera_id} ready and working - Frame: {test_frame.shape}")
    print("🎥 Starting live pixelated face protection + pose tracking")
    print("🎮 Controls: 'q' to quit, ← → arrows (or A/D keys) to switch cameras")
    
    try:
        frame_count = 0
        print("🔄 Entering main processing loop...")
        print("💡 Press 'q' to quit, ESC to exit, or close window to stop")
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to capture frame from camera {current_camera_id} at frame {frame_count}")
                print("💡 Camera may be in use by another application")
                break
            
            frame_count += 1
            if frame_count % 30 == 1:  # Log every 30 frames (roughly once per second at 30fps)
                print(f"📽️  Processing frame {frame_count} - Shape: {frame.shape}")
            
            try:
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect current faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
                
                # Process currently detected faces
                new_face_memory = []
                
                for (x, y, w, h) in current_faces:
                    # Expand area for better coverage
                    margin = 40  # Increased margin for better security
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    # Add to memory with full persistence
                    new_face_memory.append((x1, y1, x2, y2, PERSIST_FRAMES))
                    
                    # Apply heavy pixelation
                    frame = pixelate_region(frame, x1, y1, x2, y2, pixel_size=15)
                
                # Process persisted faces from memory
                for (x1, y1, x2, y2, frames_left) in face_memory:
                    if frames_left > 0:
                        # Check if this area overlaps with any current detection
                        overlaps = False
                        for (fx1, fy1, fx2, fy2, _) in new_face_memory:
                            if not (x2 < fx1 or fx2 < x1 or y2 < fy1 or fy2 < y1):
                                overlaps = True
                                break
                        
                        # If no overlap with current detection, apply persistence blur
                        if not overlaps:
                            frame = pixelate_region(frame, x1, y1, x2, y2, pixel_size=18)
                            new_face_memory.append((x1, y1, x2, y2, frames_left - 1))
                
                # Update face memory
                face_memory = new_face_memory
                
                # POSE TRACKING - Detect and draw body pose with joint angles
                frame = pose_estimator.detect_pose_and_angles(frame)
                
            except Exception as e:
                print(f"❌ Error during frame processing: {e}")
                print(f"📊 Frame info: {frame.shape if 'frame' in locals() else 'No frame'}")
                import traceback
                traceback.print_exc()
                break
            
            # Add VPU status with enhanced info
            active_faces = len([f for f in face_memory if f[4] > 0])
            cv2.putText(frame, "VPU - SECURE + POSE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Face zones: {active_faces} | Pose: SMOOTH", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Camera: {current_camera_id} ({current_camera_idx + 1}/{len(available_cameras)}) | ← → or A/D to switch", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display
            try:
                cv2.imshow('VPU - Enhanced Protection + Pose Tracking', frame)
                # Check if window was closed by user
                if cv2.getWindowProperty('VPU - Enhanced Protection + Pose Tracking', cv2.WND_PROP_VISIBLE) < 1:
                    print("🛑 Window closed by user")
                    break
            except cv2.error as e:
                print(f"❌ Display error: {e}")
                print("💡 Running in headless mode or no display available")
                break
            
            # Handle keyboard input
            try:
                key = cv2.waitKey(1) & 0xFF
            except Exception as e:
                print(f"❌ Error in waitKey: {e}")
                break
            
            # Debug key press (show ALL key events) - remove after debugging
            # if key != 255: 
            #     print(f"Key detected: {key} (char: '{chr(key) if 32 <= key <= 126 else 'N/A'}') - ord('q')={ord('q')}")
            
            # Quit on 'q' or ESC
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("🛑 User requested quit")
                break
            
            # Camera switching with arrow keys or A/D keys (multiple key codes for compatibility)
            if key in [81, 2, 84, 0x51, ord('a'), ord('A')]:  # Left arrow or A key
                if len(available_cameras) > 1:
                    current_camera_idx = (current_camera_idx - 1) % len(available_cameras)
                    new_camera_id = available_cameras[current_camera_idx]
                    
                    # Switch camera
                    cap.release()
                    cap = initialize_camera(new_camera_id)
                    if cap:
                        current_camera_id = new_camera_id
                        print(f"📹 Switched to camera {current_camera_id}")
                        # Reset face memory when switching cameras
                        face_memory = []
                    else:
                        # If new camera fails, go back to previous
                        current_camera_idx = (current_camera_idx + 1) % len(available_cameras)
                        cap = initialize_camera(available_cameras[current_camera_idx])
                        print(f"❌ Camera {new_camera_id} failed, staying on {current_camera_id}")
            
            if key in [83, 3, 82, 0x53, ord('d'), ord('D')]:  # Right arrow or D key
                if len(available_cameras) > 1:
                    current_camera_idx = (current_camera_idx + 1) % len(available_cameras)
                    new_camera_id = available_cameras[current_camera_idx]
                    
                    # Switch camera
                    cap.release()
                    cap = initialize_camera(new_camera_id)
                    if cap:
                        current_camera_id = new_camera_id
                        print(f"📹 Switched to camera {current_camera_id}")
                        # Reset face memory when switching cameras
                        face_memory = []
                    else:
                        # If new camera fails, go back to previous
                        current_camera_idx = (current_camera_idx - 1) % len(available_cameras)
                        cap = initialize_camera(available_cameras[current_camera_idx])
                        print(f"❌ Camera {new_camera_id} failed, staying on {current_camera_id}")
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🔒 VPU enhanced protection + pose tracking shutdown complete")

if __name__ == "__main__":
    main()
