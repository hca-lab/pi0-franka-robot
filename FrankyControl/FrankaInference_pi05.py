#!/usr/bin/env python3
"""
Franka Robot Inference Script for Real-Time Kernel

This script runs on the real-time kernel machine and:
1. Uses franky to read robot state (qpos + gripper)
2. Uses franky to control robot with end-effector velocity control
3. Uses RealSense cameras to capture images
4. Converts BGR to RGB
5. Connects to policy server (model runs on separate server)

Setup:
- Install: pip install pyrealsense2 franky
- Install openpi-client: cd packages/openpi-client && pip install -e .
"""
import cv2
import logging
import time
from franky import *
from typing import Optional, Tuple
# from franka_robot import FrankaRobot, RobotInputs
import numpy as np
import pyrealsense2 as rs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Policy server (model runs here)
SERVER_IP = "10.1.38.140"  # IP address of the machine running the policy server
SERVER_PORT = 8000

# RealSense cameras (on real-time kernel machine)
# IMPORTANT: Set these to specify which camera is base vs wrist
# 
# Base camera: Usually the third-person/external camera viewing the scene
# Wrist camera: Usually mounted on the robot's end-effector/wrist
#
# To find your camera serial numbers:
#   1. Run: rs-enumerate-devices
#   2. Or uncomment list_realsense_cameras() in main() and run the script
#
# Then set them here:
BASE_CAMERA_SN = "234322305598"  # Serial number of BASE camera (third-person view)
WRIST_CAMERA_SN = "241122306284"  # Serial number of WRIST camera (mounted on robot)
#
# If both are None, the code will auto-detect:
#   - First camera found = base camera
#   - Second camera found = wrist camera

# Franka robot (on real-time kernel machine)
FRANKA_IP = "10.1.38.23"  # IP address of your Franka robot

# Task instruction
TASK_INSTRUCTION = "pick up the salt box"

# Control parameters
MAX_STEPS = 200  # Maximum number of control steps
OPEN_LOOP_HORIZON = 5  # Execute N actions from chunk before querying again

# Control frequency (Hz) - How often to execute actions on the robot
# Common values:
#   - 10 Hz  = 0.1s per step  (slower, more stable)
#   - 20 Hz  = 0.05s per step (recommended, good balance)
#   - 30 Hz  = 0.033s per step (faster, requires good real-time performance)
#   - 50 Hz  = 0.02s per step  (very fast, needs excellent real-time kernel)
#
# Note: This should match your training data frequency if possible.
# Franka robots typically support 20-50 Hz control.
CONTROL_FREQUENCY = 30.0  # Control frequency in Hz (20 Hz = 0.05s per step)

# Velocity limits (adjust based on your needs)
MAX_LINEAR_VELOCITY = 1  # m/s for x, y, z
MAX_ANGULAR_VELOCITY = 1 # rad/s for roll, pitch, yaw
GRIPPER_SPEED = 0.05  # m/s
MAX_GRIPPER_WIDTH = 0.1  # meters (8cm)


# ============================================================================
# Helper Functions
# ============================================================================

def list_realsense_cameras():
    """List all available RealSense cameras and their serial numbers.
    
    Run this function to find your camera serial numbers, then set
    BASE_CAMERA_SN and WRIST_CAMERA_SN above.
    
    The function will help you identify which camera should be base vs wrist.
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        print("\n" + "=" * 70)
        print("Available RealSense Cameras:")
        print("=" * 70)
        for i, device in enumerate(devices):
            sn = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"\n  Camera {i+1}:")
            print(f"    Name: {name}")
            print(f"    Serial Number: {sn}")
            print(f"    Usage: {'BASE (third-person view)' if i == 0 else 'WRIST (robot-mounted)' if i == 1 else 'Available'}")
        
        print("\n" + "=" * 70)
        print("To configure cameras, set in the configuration section:")
        print("=" * 70)
        if len(devices) >= 1:
            print(f"\n  BASE_CAMERA_SN = \"{devices[0].get_info(rs.camera_info.serial_number)}\"  # Camera 1")
        if len(devices) >= 2:
            print(f"  WRIST_CAMERA_SN = \"{devices[1].get_info(rs.camera_info.serial_number)}\"  # Camera 2")
        print("\nNote: You can swap these if Camera 1 should be wrist and Camera 2 should be base.")
        print("=" * 70 + "\n")
        return [d.get_info(rs.camera_info.serial_number) for d in devices]
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return []


# ============================================================================
# RealSense Camera Interface
# ============================================================================

class RealSenseCameras:
    """Manages two RealSense cameras for base and wrist views."""
    
    def __init__(self, base_sn: Optional[str] = None, wrist_sn: Optional[str] = None):
        """Initialize RealSense cameras.
        
        Args:
            base_sn: Serial number of base camera (None to auto-detect)
            wrist_sn: Serial number of wrist camera (None to auto-detect)
        """
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        
        if len(devices) < 2:
            raise RuntimeError(f"Found {len(devices)} RealSense device(s), need at least 2")
        
        device_sns = [d.get_info(rs.camera_info.serial_number) for d in devices]
        logger.info(f"Found RealSense devices: {device_sns}")
        
        # Assign cameras based on configuration
        # BASE_CAMERA_SN and WRIST_CAMERA_SN determine which camera is which
        if base_sn is None:
            base_sn = device_sns[0]
            logger.info(f"Auto-detected base camera: {base_sn} (first camera found)")
        if wrist_sn is None:
            wrist_sn = device_sns[1] if len(device_sns) > 1 else device_sns[0]
            logger.info(f"Auto-detected wrist camera: {wrist_sn} (second camera found)")
        
        if base_sn not in device_sns:
            raise RuntimeError(f"Base camera SN {base_sn} not found. Available cameras: {device_sns}")
        if wrist_sn not in device_sns:
            raise RuntimeError(f"Wrist camera SN {wrist_sn} not found. Available cameras: {device_sns}")
        
        if base_sn == wrist_sn:
            raise RuntimeError(f"Base and wrist cameras cannot be the same! Both set to: {base_sn}")
        
        self.base_sn = base_sn
        self.wrist_sn = wrist_sn
        
        logger.info(f"Camera assignment:")
        logger.info(f"  BASE camera (third-person view): SN {base_sn}")
        logger.info(f"  WRIST camera (robot-mounted): SN {wrist_sn}")
        
        # Configure and start pipelines
        self.base_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()
        
        base_config = rs.config()
        base_config.enable_device(base_sn)
        base_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        wrist_config = rs.config()
        wrist_config.enable_device(wrist_sn)
        wrist_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        logger.info(f"Starting base camera (SN: {base_sn})")
        self.base_pipeline.start(base_config)
        
        logger.info(f"Starting wrist camera (SN: {wrist_sn})")
        self.wrist_pipeline.start(wrist_config)
        
        # Warm up cameras
        logger.info("Warming up cameras...")
        for _ in range(10):
            self.base_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()
        logger.info("Cameras ready!")
    
    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture images from both cameras.
        
        Returns:
            Tuple of (base_image, wrist_image) as numpy arrays (H, W, 3) in RGB format.
            RealSense outputs BGR, so we convert to RGB here.
        """
        # Get frames
        base_frames = self.base_pipeline.wait_for_frames()
        wrist_frames = self.wrist_pipeline.wait_for_frames()
        
        # Extract color frames
        base_color_frame = base_frames.get_color_frame()
        wrist_color_frame = wrist_frames.get_color_frame()
        if not base_color_frame or not wrist_color_frame:
            raise RuntimeError("Failed to get color frames")
        
        # Convert to numpy arrays (RealSense outputs BGR format)
        base_image = np.asanyarray(base_color_frame.get_data())
        wrist_image = np.asanyarray(wrist_color_frame.get_data())
        
        # Convert BGR to RGB (RealSense outputs BGR, but model expects RGB)
        base_image = base_image[..., ::-1]
        wrist_image = wrist_image[..., ::-1]
        cv2.imshow("Base Camera", base_image[..., ::-1])
        cv2.imshow("Wrist Camera", wrist_image[..., ::-1])
        cv2.waitKey(1)  # important for window refresh
        return base_image, wrist_image
    
    def stop(self):
        """Stop camera pipelines."""
        self.base_pipeline.stop()
        self.wrist_pipeline.stop()
        logger.info("Cameras stopped")


# ============================================================================
# Franka Robot Interface using franky
# ============================================================================

class FrankaRobot:
    """Manages Franka robot using franky library.
    
    - Reads robot state (qpos + gripper) using franky
    - Controls robot with end-effector velocity control using franky
    """
    
    def __init__(self, robot_ip: str):
        """Initialize Franka robot connection.
        
        Args:
            robot_ip: IP address of the Franka robot
        """
        try:
            import franky
            self.franky = franky
            logger.info("Using franky for Franka control")
            
            # Initialize robot and gripper
            self.robot = franky.Robot(robot_ip)
            self.gripper = franky.Gripper(robot_ip)
            self.gripper_close = False
            self.close_gripper = False
            logger.info(f"Connected to Franka robot at {robot_ip}")
        except ImportError:
            raise RuntimeError(
                "franky library not found. Install it with: pip install franky"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Franka robot at {robot_ip}: {e}")
    
    def get_state(self) -> np.ndarray:
        """Get current robot state using franky.
        
        Returns:
            Robot state as numpy array: [qpos (7), gripper_width (1)]
            - qpos: 7 joint positions
            - gripper_width: gripper opening width
        """
        # Use franky to read robot state
        state = self.robot.state
        qpos = np.array(self.robot.current_joint_state.position)  # 7 joint positions
        gripper_width = np.array([self.gripper.width])  # Gripper width
        # print(qpos)
        # print(gripper_width)
        return np.concatenate([qpos, gripper_width]).astype(np.float32)
    
    def execute_action(self, action: np.ndarray, dt: float = 0.05):
        """Execute action on robot using franky end-effector velocity control.
        
        Args:
            action: Action array of shape (7,): [x_vel, y_vel, z_vel, roll_vel, pitch_vel, yaw_vel, gripper]
                   All velocities are in normalized [-1, 1] range
            dt: Time step for execution (default 0.05s for 20Hz)
        """
        # if len(action) < 8:
        #     raise ValueError(f"Action must have 7 dimensions, got {len(action)}")
        
        # Parse action: [x, y, z, roll, pitch, yaw velocities, gripper]
        # detla_joint_angle = action[:7]  # End-effector velocities
        # gripper_action = action[7] # Gripper command
        ee_velocities = action[:6]
        gripper_action = action[6]
        # # Clip actions to safe ranges
        # ee_velocities = np.clip(ee_velocities, -1.0, 1.0)
        # gripper_action = np.clip(gripper_action, -1.0, 1.0)
        
        # Scale from normalized [-1, 1] to actual velocities
        linear_vel = ee_velocities[:3] * MAX_LINEAR_VELOCITY  # x, y, z in m/s
        angular_vel = ee_velocities[3:6] * MAX_ANGULAR_VELOCITY  # roll, pitch, yaw in rad/s
        # # linear_vel = ee_velocities[:3]  # x, y, z in m/s
        # # angular_vel = ee_velocities[3:6]  # roll, pitch, yaw in rad/s
        # angular_vel[1] = 0
        # angular_vel[2] = 0
        # Combine into 6D twist vector [vx, vy, vz, wx, wy, wz]
        twist = np.concatenate([linear_vel, angular_vel])
        
        vel = RobotVelocity(Twist(twist[:3],twist[3:]),)
        # Execute end-effector velocity control using franky
        try:
            # detla_joint_angle = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            # motion = self.franky.JointMotion(detla_joint_angle, ReferenceType.Relative, return_when_finished=True)
            # # motion = self.franky.CartesianMotion(Affine(linear_vel), ReferenceType.Relative)
            motion = self.franky.CartesianVelocityMotion(vel)
            robot.relative_dynamics_factor = 0.1
            self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.1, 0.15)
            self.robot.move(motion, asynchronous=True)
            # self.robot.move(motion)
            time.sleep(0.001)
        except Exception as e:
            logger.error(f"Cartesian velocity control failed: {e}")
            raise RuntimeError(f"Failed to execute Cartesian velocity control: {e}")
        
        # Execute gripper control using franky
        # Map gripper action from [-1, 1] to [0, max_width]
        gripper_width = ((gripper_action + 1.0) / 2.0) * MAX_GRIPPER_WIDTH
        if gripper_action < -0.003:
            self.close_gripper = True
        if self.close_gripper == True and self.gripper_close == False:
            gripper_width = 0
            try:
                # Use async gripper control to avoid blocking
                self.gripper.grasp(width=0.0, speed=GRIPPER_SPEED, force=50, epsilon_outer=1.0)
                self.gripper_close = True
            except Exception as e:
                logger.warning(f"Async gripper control failed, trying sync: {e}")
                try:
                    self.gripper.move(gripper_width, GRIPPER_SPEED)
                except Exception as e2:
                    logger.error(f"Gripper control failed: {e2}")
        else: gripper_width = MAX_GRIPPER_WIDTH   

    
    def stop(self):
        """Stop robot and gripper."""
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception as e:
                logger.warning(f"Error stopping robot: {e}")
        if hasattr(self, 'gripper') and self.gripper is not None:
            try:
                self.gripper.stop()
            except Exception as e:
                logger.warning(f"Error stopping gripper: {e}")
        logger.info("Robot stopped")


# ============================================================================
# Main Inference Loop
# ============================================================================

def main():
    """Main inference loop running on real-time kernel."""
    
    logger.info("=" * 60)
    logger.info("Franka Robot Inference - Real-Time Kernel")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # 1. Initialize RealSense cameras
    cameras = RealSenseCameras(BASE_CAMERA_SN, WRIST_CAMERA_SN)
    
    # 2. Initialize Franka robot (using franky)
    robot = FrankaRobot(FRANKA_IP)
    
    # 3. Initialize policy client (connects to server)
    logger.info("=" * 60)
    logger.info(f"Connecting to policy server at {SERVER_IP}:{SERVER_PORT}")
    logger.info("=" * 60)
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_IP, port=SERVER_PORT)
        logger.info("✓ WebSocket client created")
    except Exception as e:
        logger.error(f"✗ Failed to create WebSocket client: {e}")
        raise
    
    # Verify connection and get server metadata
    try:
        logger.info("Requesting server metadata...")
        metadata = client.get_server_metadata()
        logger.info("✓ Successfully connected to policy server!")
        logger.info(f"Server metadata: {metadata}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Failed to connect to policy server: {e}")
        logger.error("Make sure the policy server is running on the server machine.")
        logger.error(f"  Server IP: {SERVER_IP}")
        logger.error(f"  Server Port: {SERVER_PORT}")
        raise
    
    # Control variables
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    step_duration = 1.0 / CONTROL_FREQUENCY

    last_send_time = 0.0
    
    logger.info(f"Starting inference loop")
    logger.info(f"Task: '{TASK_INSTRUCTION}'")
    logger.info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
    logger.info(f"Open loop horizon: {OPEN_LOOP_HORIZON}")
    logger.info("-" * 60)
    logger.info("Server Communication:")
    logger.info(f"  - Server IP: {SERVER_IP}")
    logger.info(f"  - Server Port: {SERVER_PORT}")
    logger.info(f"  - Query server every {OPEN_LOOP_HORIZON} steps")
    logger.info("-" * 60)

    try:
        for step in range(MAX_STEPS):

            step_start_time = time.time()
            

            base_image, wrist_image = cameras.get_images()
            
            # 2. Get robot state using franky (qpos + gripper)
            robot_state = robot.get_state()
            
            # 3. Query policy server if needed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= OPEN_LOOP_HORIZON:
                actions_from_chunk_completed = 0
                
            # Prepare observation
            # logger.debug(f"Step {step}: Preparing observation for server...")
            # Images are already RGB (converted from BGR in get_images())
            observation = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(base_image, 224, 224)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_image, 224, 224)
                ),
                "observation/state": robot_state,  # qpos + gripper (unnormalized, server handles normalization)
                "prompt": TASK_INSTRUCTION,
            }
            
            # # Send observation to policy server and get action chunk
            # logger.info(f"Step {step}: Sending observation to server at {SERVER_IP}:{SERVER_PORT}...")
            logger.debug(f"  - Image shape: {observation['observation/image'].shape}")
            logger.debug(f"  - Wrist image shape: {observation['observation/wrist_image'].shape}")
            logger.debug(f"  - State shape: {observation['observation/state'].shape}")
            logger.debug(f"  - Prompt: '{observation['prompt']}'")
            
            # Expected: [x_vel, y_vel, z_vel, roll_vel, pitch_vel, yaw_vel, gripper]
            # Shape: (action_horizon, 7) where action_horizon is typically 10
            inference_start = time.time()
            try:
                result = client.infer(observation)
                print(result)
                inference_time = (time.time() - inference_start) * 1000  # ms
                logger.info(f"✓ Received response from server (took {inference_time:.1f}ms)")
            except Exception as e:
                logger.error(f"✗ Failed to get response from server: {e}")
                raise
            
            pred_action_chunk = result["actions"]
            
            # Validate action dimensions
            if pred_action_chunk.shape[1] != 8:
                logger.warning(
                    f"⚠ Expected action dimension 8 (x,y,z,roll,pitch,yaw,gripper), "
                    f"got {pred_action_chunk.shape[1]}. Action shape: {pred_action_chunk.shape}"
                )
            else:
                logger.info(
                    f"✓ Action chunk received: shape {pred_action_chunk.shape}, "
                    f"inference time: {inference_time:.1f}ms"
                )
            
            # 4. Select current action from chunk
            # print(pred_action_chunk)
            action = pred_action_chunk[actions_from_chunk_completed].copy()
            actions_from_chunk_completed += 1
            # print(action)
            # 5. Execute action on robot using franky (end-effector velocity control)
            # print(action)
            
            # now = time.time()
            # if now - last_send_time >= 0.2:
            robot.execute_action(action, dt=step_duration)
                # last_send_time = now
            
            # 6. Maintain control frequency
            elapsed = time.time() - step_start_time
            sleep_time = max(0, step_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if step % 50 == 0:
                logger.info(f"Step {step}/{MAX_STEPS} completed")
                logger.info(f"  - Server queries: {step // OPEN_LOOP_HORIZON + 1}")
                logger.info(f"  - Actions executed: {step}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
    finally:
        logger.info("Stopping components...")
        cameras.stop()
        robot.stop()
        logger.info("Inference loop ended")


if __name__ == "__main__":
    # Uncomment the line below to list available cameras and their serial numbers
    # list_realsense_cameras()
    # exit()
    
    main()
