import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import mujoco
import numpy as np
import rerun as rr
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial.transform import Rotation as R

from robot import create_so101, lerobot_FK, lerobot_IK, manipulability, return_jacobian
from urdf_utils import init_rerun_with_urdf, log_joint_angles

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# ------------------------------
# MuJoCo / Robot setup
# ------------------------------
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
mjdata = mujoco.MjData(mjmodel)

qpos_indices = np.array(
    [mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES]
)
robot = create_so101()

control_qlimit = [
    [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
    [2.1, 0.0, 3.1, 1.475, 3.1, 1.5],
]
control_glimit = [
    [0.125, -0.4, 0.046, -3.1, -0.75, -1.5],
    [0.340, 0.4, 0.23, 2.0, 1.57, 1.5],
]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
command_qpos = init_qpos.copy()

init_pose = lerobot_FK(init_qpos[1:5], robot=robot)
command_pose = init_pose.copy()

# ------------------------------
# MediaPipe setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
    max_num_hands=1,
)

# ------------------------------
# Rerun setup
# ------------------------------
timestamp = int(time.time())
urdf_path = Path("so101/so101.urdf")
robot_name, joint_paths = init_rerun_with_urdf(
    f"so101_hand_tracking_{timestamp}", urdf_path
)


# ------------------------------
# Helper utilities
# ------------------------------
def clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def map_range_clamped(
    value: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    if np.isclose(in_max - in_min, 0.0):
        return (out_min + out_max) * 0.5
    t = (value - in_min) / (in_max - in_min)
    t = np.clip(t, 0.0, 1.0)
    return out_min + t * (out_max - out_min)


def smooth_value(current: float, target: float, alpha: float) -> float:
    return float(current + alpha * (target - current))


def landmark_vec(lm: landmark_pb2.NormalizedLandmark) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def compute_hand_metrics(
    landmark_list: landmark_pb2.NormalizedLandmarkList,
) -> Dict[str, np.ndarray]:
    idx = mp_hands.HandLandmark

    wrist = landmark_vec(landmark_list.landmark[idx.WRIST])
    thumb_cmc = landmark_vec(landmark_list.landmark[idx.THUMB_CMC])
    index_mcp = landmark_vec(landmark_list.landmark[idx.INDEX_FINGER_MCP])
    middle_mcp = landmark_vec(landmark_list.landmark[idx.MIDDLE_FINGER_MCP])
    ring_mcp = landmark_vec(landmark_list.landmark[idx.RING_FINGER_MCP])
    pinky_mcp = landmark_vec(landmark_list.landmark[idx.PINKY_MCP])
    index_tip = landmark_vec(landmark_list.landmark[idx.INDEX_FINGER_TIP])
    thumb_tip = landmark_vec(landmark_list.landmark[idx.THUMB_TIP])

    palm_core = np.vstack(
        [wrist, thumb_cmc, index_mcp, middle_mcp, ring_mcp, pinky_mcp]
    )
    center = palm_core.mean(axis=0)

    # Use strict 2D distance for span/length to avoid Z noise
    span = np.linalg.norm(index_mcp[:2] - pinky_mcp[:2])
    length = np.linalg.norm(middle_mcp[:2] - wrist[:2])
    pinch = np.linalg.norm(index_tip - thumb_tip)

    # Construct Basis Vectors
    # Primary Axis: Wrist to Middle Finger (Y)
    y_vec = middle_mcp - wrist
    # Secondary Axis: Pinky to Index (X)
    x_vec = index_mcp - pinky_mcp

    rotation = None
    if np.linalg.norm(x_vec) > 1e-6 and np.linalg.norm(y_vec) > 1e-6:
        y_axis = y_vec / np.linalg.norm(y_vec)
        x_axis = x_vec / np.linalg.norm(x_vec)

        # Z = X cross Y (Normal to palm)
        z_axis = np.cross(x_axis, y_axis)
        if np.linalg.norm(z_axis) > 1e-6:
            z_axis = z_axis / np.linalg.norm(z_axis)
            # Re-orthogonalize X to ensure perfect rotation matrix
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            # Matrix columns [X, Y, Z]
            rotation = np.column_stack((x_axis, y_axis, z_axis))

    return {
        "center": center,
        "span": span,
        "length": length,
        "pinch": pinch,
        "rotation": rotation,
        "wrist": wrist,
    }


# ------------------------------
# Control parameters
# ------------------------------
BASE_GAIN = 4.8
HEIGHT_GAIN = 0.42

# CHANGED: High negative gain.
# Hand gets bigger (closer to cam) -> Delta Span positive -> Radius decreases (retract)
# Hand gets smaller (further from cam) -> Delta Span negative -> Radius increases (extend)
SPAN_GAIN = -1.2

# Increased orientation gain to make wrist responsive
ORIENTATION_GAIN = np.array([1.8, 1.8, 1.8])

NO_HAND_TIMEOUT = 0.65
SMOOTH_TRACK = {
    "base": 0.28,
    "radius": 0.30,
    "height": 0.30,
    "orientation": 0.20,
    "gripper": 0.35,
}
SMOOTH_RECENTER = 0.06

calibration: Optional[Dict[str, np.ndarray]] = None
baseline_rotation: Optional[R] = None
last_detection_time = -np.inf

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open default camera device.")

print("Starting Hand Tracking Control...")
print("Raise a neutral hand pose inside the camera frame to capture calibration.")
print("Controls:")
print("  • Move hand left/right to rotate base.")
print("  • Move hand closer/farther (size change) to extend/retract.")
print("  • Raise/lower palm height to adjust end-effector height.")
print("  • Twist and bend palm to adjust wrist orientation.")
print("  • Pinch thumb and index to close the gripper.")
print("Hotkeys: 'r' to recalibrate, 'q' to quit.")

try:
    while cap.isOpened():
        step_start = time.time()

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # mediapipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        detection = None
        hand_landmarks_proto = None

        if results.multi_hand_landmarks:
            hand_landmarks_proto = results.multi_hand_landmarks[0]
            detection = compute_hand_metrics(hand_landmarks_proto)

        # control logic
        if detection and detection["rotation"] is not None:
            if calibration is None:
                calibration = {
                    "center": detection["center"].copy(),
                    "span": detection["span"],
                    "length": detection["length"],
                }
                baseline_rotation = R.from_matrix(detection["rotation"])
                last_detection_time = time.time()
                print("Calibration captured. You can now control the arm.")
            else:
                last_detection_time = time.time()

            if baseline_rotation is None:
                baseline_rotation = R.from_matrix(detection["rotation"])

            # Calculate Deltas
            delta_center = detection["center"] - calibration["center"]
            delta_height = calibration["center"][1] - detection["center"][1]
            delta_span = detection["span"] - calibration["span"]

            # 1. Base (Rotation)
            base_target = init_qpos[0] - delta_center[0] * BASE_GAIN
            base_target = clamp(base_target, control_qlimit[0][0], control_qlimit[1][0])
            command_qpos[0] = smooth_value(
                command_qpos[0], base_target, SMOOTH_TRACK["base"]
            )

            # 2. Radius (Span/Depth)
            radius_shift = delta_span * SPAN_GAIN
            radius_target = init_pose[0] + radius_shift
            radius_target = clamp(
                radius_target, control_glimit[0][0], control_glimit[1][0]
            )
            command_pose[0] = smooth_value(
                command_pose[0], radius_target, SMOOTH_TRACK["radius"]
            )

            # 3. Height
            height_shift = delta_height * HEIGHT_GAIN
            height_target = init_pose[2] + height_shift
            height_target = clamp(
                height_target, control_glimit[0][2], control_glimit[1][2]
            )
            command_pose[2] = smooth_value(
                command_pose[2], height_target, SMOOTH_TRACK["height"]
            )

            # 4. Orientation (Wrist)
            current_rot = R.from_matrix(detection["rotation"])
            relative_rot = baseline_rotation.inv() * current_rot
            relative_euler = relative_rot.as_euler("xyz", degrees=False)

            orientation_target = init_pose[3:] + relative_euler * ORIENTATION_GAIN
            orientation_target = np.clip(
                orientation_target,
                np.array(control_glimit[0][3:6]),
                np.array(control_glimit[1][3:6]),
            )
            command_pose[3:] = command_pose[3:] + SMOOTH_TRACK["orientation"] * (
                orientation_target - command_pose[3:]
            )

            # 5. Gripper
            pinch_ratio = detection["pinch"] / max(detection["span"], 1e-5)
            gripper_target = map_range_clamped(
                pinch_ratio, 0.18, 0.45, control_qlimit[0][5], control_qlimit[1][5]
            )
            command_qpos[5] = smooth_value(
                command_qpos[5], gripper_target, SMOOTH_TRACK["gripper"]
            )

            # Visual Overlay
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            cv2.putText(
                image_bgr,
                "TRACKING",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            # Timeout / Recenter Logic
            elapsed = time.time() - last_detection_time
            if elapsed > NO_HAND_TIMEOUT:
                command_qpos[0] = smooth_value(
                    command_qpos[0], init_qpos[0], SMOOTH_RECENTER
                )
                command_pose[:3] = command_pose[:3] + SMOOTH_RECENTER * (
                    init_pose[:3] - command_pose[:3]
                )
                command_pose[3:] = command_pose[3:] + SMOOTH_RECENTER * (
                    init_pose[3:] - command_pose[3:]
                )
                command_qpos[5] = smooth_value(
                    command_qpos[5], init_qpos[5], SMOOTH_RECENTER
                )

            cv2.putText(
                image_bgr,
                "WAITING FOR HAND",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # inverse kinematics
        command_pose[1] = 0.0  # Planar constraint
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        ik_target = command_pose.copy()

        # Solve IK
        qpos_inv, ik_success = lerobot_IK(fd_qpos, ik_target, robot=robot)

        # Calculate Manipulability
        jacobian = return_jacobian(fd_qpos, robot=robot)
        m_value, condition = manipulability(jacobian)

        if ik_success:
            command_qpos[1:5] = qpos_inv[:4]
            mjdata.qpos[qpos_indices] = command_qpos
            mujoco.mj_step(mjmodel, mjdata)  # Physics step updates xanchor
        else:
            # Safety fallback
            command_pose[:3] = command_pose[:3] + 0.15 * (
                init_pose[:3] - command_pose[:3]
            )

        # Log Target
        radius = command_pose[0]
        base_angle = command_qpos[0]
        ee_target_pos = np.array(
            [radius * np.cos(base_angle), radius * np.sin(base_angle), command_pose[2]]
        )
        rr.log(
            "end_effector/target",
            rr.Points3D([ee_target_pos], radii=0.012, colors=[0, 220, 255]),
        )

        # Log Robot Chain (Using MuJoCo's internal state, not manual FK)
        chain_positions = []
        # Extract the 3D position of every joint pivot directly from simulation data
        for name in JOINT_NAMES:
            jid = mjmodel.joint(name).id
            chain_positions.append(mjdata.xanchor[jid])

        # Cast to numpy for Rerun
        chain_positions = np.array(chain_positions, dtype=np.float64)

        rr.log(
            "end_effector/arm_chain",
            rr.LineStrips3D([chain_positions], colors=[255, 180, 0], radii=0.006),
        )
        rr.log(
            "end_effector/joints",
            rr.Points3D(chain_positions, radii=0.009, colors=[255, 0, 0]),
        )
        rr.log("manipulability/value", rr.Scalars(m_value))

        log_joint_angles(robot_name, joint_paths, JOINT_NAMES, command_qpos[:6])

        cv2.imshow("SO101 Hand Tracking", image_bgr)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            calibration = None
            baseline_rotation = None
            print("Recalibrating...")

        time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
