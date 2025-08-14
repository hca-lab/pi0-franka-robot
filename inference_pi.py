import pyrealsense2 as rs
import cv2
import numpy as np
from PIL import Image

from openpi.training import config
from openpi.policies import policy_config


import torch
import numpy as np
import time

import socket
import struct

# =========================
# Socket Communication Setup (optional)
# =========================
# ---- Actions sender (PC1 -> PC2) ----
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('10.1.38.22', 9090)   # PC2 receiver for actions
ACTION_FMT = "<8d"

# ---- State receiver (PC2 -> PC1) ----
STATE_FMT = "14d"                       # x,y,z,roll,pitch,yaw,gripper (float64)
STATE_LISTEN_IP = "0.0.0.0"             # listen on all interfaces
STATE_LISTEN_PORT = 9091                # <-- use a different port than 9090
state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
state_sock.bind((STATE_LISTEN_IP, STATE_LISTEN_PORT))
state_sock.settimeout(0.0)              # non-blocking
last_robot_state = np.zeros(14, dtype=np.float64)



def crop_and_resize_cv(image_pil: Image.Image, crop_scale: float = 0.9, out_hw=(224, 224)) -> Image.Image:
    """Center-crop by area ratio then resize, using NumPy/OpenCV only."""
    img = np.array(image_pil)  # RGB uint8, (H,W,C)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3)")
    H, W = img.shape[:2]

    s = float(crop_scale) ** 0.5
    crop_h = max(1, int(round(H * s)))
    crop_w = max(1, int(round(W * s)))

    y0 = max((H - crop_h) // 2, 0); y1 = y0 + crop_h
    x0 = max((W - crop_w) // 2, 0); x1 = x0 + crop_w
    img_c = img[y0:y1, x0:x1]

    out_w, out_h = out_hw[1], out_hw[0]  # cv2 takes (W,H)
    img_r = cv2.resize(img_c, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_r, mode="RGB")

def preprocess_image(image_pil: Image.Image, crop_scale: float = 0.9, out_size=(224, 224)) -> Image.Image:
    """Center-crop by area 'crop_scale' and resize to out_size using OpenCV."""
    img = np.array(image_pil)  # RGB
    H, W = img.shape[:2]
    s = float(crop_scale) ** 0.5
    crop_h, crop_w = int(round(H * s)), int(round(W * s))
    y0 = max((H - crop_h) // 2, 0)
    x0 = max((W - crop_w) // 2, 0)
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    img_c = img[y0:y1, x0:x1]
    img_r = cv2.resize(img_c, out_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_r, mode="RGB")



def poll_robot_state_nonblocking():
    global last_robot_state
    while True:
        try:
            data, _ = state_sock.recvfrom(1024)
        except BlockingIOError:
            break
        except Exception:
            break
        if len(data) == struct.calcsize(STATE_FMT):
            last_robot_state = np.array(struct.unpack(STATE_FMT, data), dtype=np.float64)
    return last_robot_state




# ====== RealSense capture setup ======
def start_pipeline(serial: str):
    """Start a RealSense pipeline for a given device serial number."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    return pipeline

# Find first two connected RealSense devices
ctx = rs.context()
devices = list(ctx.query_devices())
if len(devices) < 2:
    raise RuntimeError(f"At least two RealSense D455 cameras are required, found {len(devices)}.")

serials = [d.get_info(rs.camera_info.serial_number) for d in devices[:2]]
print(f"[RealSense] Using devices: {serials[0]} (Camera 1), {serials[1]} (Camera 2)")
print("Press 'q' in any OpenCV window to quit.")

pipe1 = start_pipeline(serials[0])
pipe2 = start_pipeline(serials[1])




#####  pi0 model #####################
checkpoint_path = "/home/yi/ModelCheckPoints/VLA-Model/openpi-assets/checkpoints/pi0_fast_droid"
cfg = config.get_config("pi0_fast_droid")
policy = policy_config.create_trained_policy(cfg, checkpoint_path)



send_interval = 0.2  # seconds between sends (5 Hz)
last_send_time = 0.0

i = 0

try:
    while True:
        frames1 = pipe1.wait_for_frames()
        frames2 = pipe2.wait_for_frames()
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()
        if not color_frame1 or not color_frame2:
            continue

        img1 = np.asanyarray(color_frame1.get_data())  # BGR
        img2 = np.asanyarray(color_frame2.get_data())  # BGR
        pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        wrist_image = preprocess_image(pil1)
        front_image = preprocess_image(pil2)
        disp1 = cv2.cvtColor(np.array(wrist_image), cv2.COLOR_RGB2BGR)
        disp2 = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)

        # Resize for larger display (e.g., 3x larger)
        display_scale = 3
        disp1_large = cv2.resize(img1, (224 * display_scale, 224 * display_scale), interpolation=cv2.INTER_NEAREST)
        disp2_large = cv2.resize(img2, (224 * display_scale, 224 * display_scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Camera 1 Processed", disp1_large)
        cv2.imshow("Camera 2 Processed", disp2_large)

        # wrist_image = wrist_image.rotate(180, expand=True)  # for PIL


        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
        
        # Only send and run inference every `send_interval` seconds
        now = time.time()
        if now - last_send_time >= send_interval:
                # # 1) Get latest robot state from PC2
                cur_robot_state = poll_robot_state_nonblocking().astype(np.float32)  # shape (14,)

                # print(f"[PC1] Current robot state: {cur_robot_state}"

                example = {
                        "observation/exterior_image_1_left": front_image,
                        "observation/wrist_image_left": wrist_image,
                        "observation/joint_position": cur_robot_state[7:14],
                        "observation/gripper_position": np.array([cur_robot_state[6]]),
                        "prompt": "pick up the orange and place it on the white plate",
                    }
                

                action_chunk = policy.infer(example)["actions"]

                action_prediction = action_chunk[0]  # Assuming the first action is the one we want

                # 3) Send to PC2 (same format as PC2 expects)
                message_UDP = struct.pack(ACTION_FMT, *action_prediction)
                sock.sendto(message_UDP, server_address)
                print(f"[PC1] Sent action to PC2: {action_prediction}")

                last_send_time = now



finally:
    pipe1.stop()
    pipe2.stop()
    cv2.destroyAllWindows()
