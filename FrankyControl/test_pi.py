#!/usr/bin/env python3
import socket, struct, time, argparse, math
import numpy as np
from typing import Tuple

from scipy.spatial.transform import Rotation

from franky import *
import franky

# =========================
# Robot setup
# =========================
ROBOT_IP = "10.1.38.23"
robot = Robot(ROBOT_IP)
robot.relative_dynamics_factor = 0.01
gripper = franky.Gripper(ROBOT_IP)


speed = 0.02  # [m/s]
force = 20.0  # [N]


q = np.array([0.31497584, -0.00210619, 0.44722789, 0.0, 0.0, 0.0])  # Initial position and orientation
quat = np.array([ 0.99416457,  0.0590707 ,  0.08990171, -0.00807068])  # Example quaternion, adjust as needed



# =========================
# Action receiver
# =========================
def make_socket(bind_ip: str, bind_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip, bind_port))
    sock.settimeout(0.1)
    return sock

# =========================
# Pose -> RPY helpers
# =========================
def quat_to_rpy(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    # ZYX (yaw-pitch-roll) convention
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(t0, t1)

    t2 = +2.0*(w*y - z*x)
    t2 =  1.0 if t2 >  1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def read_ee_pose_rpy(robot: Robot) -> Tuple[float, float, float, float, float, float]:
    cart = robot.current_cartesian_state
    pose = cart.pose.end_effector_pose
    x, y, z = pose.translation
    w, xq, yq, zq = pose.quaternion
    r, p, yaw = quat_to_rpy(w, xq, yq, zq)
    return x, y, z, r, p, yaw


def read_joint_positions(robot: Robot)-> Tuple[float, float, float, float, float, float]:
    """
    Returns a tuple of 7 joint positions (q1..q7) in radians.
    """
    joint_state = robot.current_joint_state            # franky.RobotState
    joint_pos = joint_state.position
    q0 = joint_pos[0]  # q1
    q1 = joint_pos[1]  # q2
    q2 = joint_pos[2]  # q3
    q3 = joint_pos[3]  # q4
    q4 = joint_pos[4]  # q5
    q5 = joint_pos[5]  # q6
    q6 = joint_pos[6]  # q7
    return q0, q1, q2, q3, q4, q5, q6


# Get the robot's joint state
joint_state = robot.current_joint_state
joint_pos = joint_state.position


def read_gripper_open(gripper: franky.Gripper) -> float:
    """
    Pi0 convention: 0.0 = fully open, 1.0 = fully closed.
    """
    try:
        width = float(gripper.width)   # meters
        MAX_OPEN = 0.08
        MIN_OPEN = 0.0
        width = max(MIN_OPEN, min(MAX_OPEN, width))
        closed_ratio = 1.0 - (width - MIN_OPEN) / (MAX_OPEN - MIN_OPEN)
        return closed_ratio
    except Exception as e:
        print("[gripper] read error:", e)
        return 0.0  



# =========================
# State publisher (NEW)
# =========================
PC1_IP        = "10.1.38.195"   # <-- CHANGE to your GPU/VLA PC IP
STATE_DST     = (PC1_IP, 9091)  # Port for state stream to PC1
STATE_FMT     = "<14d"          # x,y,z,r,p,yaw,g, q1..q7  (float64 x14)
STATE_BYTES   = struct.calcsize(STATE_FMT)
STATE_RATE_HZ = 50.0
STATE_PERIOD  = 1.0 / STATE_RATE_HZ

state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_state_once():
    x, y, z, r, p, yaw = read_ee_pose_rpy(robot)
    g = read_gripper_open(gripper)
    q1, q2, q3, q4, q5, q6, q7 = read_joint_positions(robot)  # radians
    pkt = struct.pack(STATE_FMT, x, y, z, r, p, yaw, g,
                                   q1, q2, q3, q4, q5, q6, q7)
    state_sock.sendto(pkt, STATE_DST)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="UDP receiver for 8x float64 action packets; publishes EE pose.")
    parser.add_argument("--ip",   type=str, default="0.0.0.0", help="Listen IP for actions (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9090,      help="Listen port for actions (default: 9090)")
    parser.add_argument("--quiet", action="store_true",        help="Only print errors")
    args = parser.parse_args()


    initial_motion = CartesianMotion(Affine([q[0], q[1], q[2]],quat), ReferenceType.Absolute)     
    robot.move(initial_motion)

    gripper.open(speed)


    # Initial absolute cartesian move (optional; adjust to your safe start)
    try:
        cart = robot.current_cartesian_state
        pose = cart.pose.end_effector_pose
        init_xyz = list(pose.translation)            # keep current XYZ
        init_quat = list(pose.quaternion)            # keep current orientation
        robot.move(CartesianMotion(Affine(init_xyz, init_quat), ReferenceType.Absolute))
    except Exception as e:
        print("[init] Skipping initial move:", e)

    # Action receiver socket
    bind_ip, bind_port = args.ip, args.port
    sock = make_socket(bind_ip, bind_port)
    print(f"[recv] Listening on {bind_ip}:{bind_port} for actions (<8d) "
          f"and publishing pose to {STATE_DST[0]}:{STATE_DST[1]} as <8d>")

    FMT_ACT   = "<8d"   # must match PC1 sender (your current setup)
    ACT_BYTES = struct.calcsize(FMT_ACT)

    # Rate control for state publisher
    last_state_send = 0.0

    # Rate control for state publisher
    last_state_send = 0.0

    # --- add: keep state across loop ---
    last_gripper_mode = None   # "open" or "close"
    last_grip_time = 0.0
    GRIPPER_COOLDOWN = 0.5


    try:
        while True:
            # ---- 1) Receive latest action (non-blocking-ish) ----
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                data = None

            if data:
                if len(data) < ACT_BYTES:
                    if not args.quiet:
                        print(f"[warn] short action packet {len(data)}B from {addr}, expected {ACT_BYTES}B")
                else:
                    act = struct.unpack(FMT_ACT, data[:ACT_BYTES])  # 8 floats from PC1

                    dq1, dq2, dq3, dq4, dq5, dq6, dq7 = 0.08*act[0], 0.08*act[1], 0.08*act[2], 0.08*act[3], 0.08*act[4], 0.08*act[5], 0.08*act[6]

                    # ---- 1) Move robot (relative joint position) ----
                    try:

                        robot.move(JointMotion([dq1, dq2, dq3, dq4, dq5, dq6, dq7], ReferenceType.Relative))
                        print(f"[move] delta joint positions: {dq1}, {dq2}, {dq3}, {dq4}, {dq5}, {dq6}, {dq7}, act[7]: {act[7]}")
                        time.sleep(0.03)  # small settle
                    except Exception as e:
                        print("[move] error:", e)


                    # ---- 2) Gripper control (only send if mode changes) ----
                    try:
                        # Pi0: 0=open, 1=close 
                        if act[7] <= 0.0:
                            current_mode = "open"
                        elif act[7] >= 1.0:
                            current_mode = "close"
                        else:
                            current_mode = last_gripper_mode  # 

                        now = time.time()
                        if current_mode != last_gripper_mode and current_mode is not None and (now - last_grip_time) >= GRIPPER_COOLDOWN:
                            if current_mode == "close":
                                success = gripper.move(0.05, speed)
                                success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0)
                                print("[gripper] Close command sent")
                            else:  # "open"
                                gripper.open(speed)
                                print("[gripper] Open command sent")

                            last_gripper_mode = current_mode
                            last_grip_time = now
                    except Exception as e:
                        print("[gripper] error:", e)


            # ---- 2) Publish current EE pose (x,y,z,r,p,y,grip) at fixed rate ----
            now = time.time()
            if now - last_state_send >= STATE_PERIOD:
                try:
                    send_state_once()
                except Exception as e:
                    print("[state] send error:", e)
                last_state_send = now

    except KeyboardInterrupt:
        print("\n[recv] Ctrl+C received, exiting...")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        print("[recv] Closed.")

if __name__ == "__main__":
    main()