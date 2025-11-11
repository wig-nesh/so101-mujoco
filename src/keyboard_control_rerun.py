import os
import mujoco
import numpy as np
import time
import threading
from pynput import keyboard
import rerun as rr
from rerun import RotationAxisAngle
from pathlib import Path
import xml.etree.ElementTree as ET

from robot import lerobot_IK, lerobot_FK, create_so101, return_jacobian, manipulability

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# ------------------------------
# Setup MuJoCo model and robot
# ------------------------------
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
mjdata = mujoco.MjData(mjmodel)

qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
robot = create_so101()

# ------------------------------
# Constants
# ------------------------------
JOINT_INCREMENT = 0.005
POSITION_INCREMENT = 0.0008

control_qlimit = [[-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
                  [ 2.1,  0.0,  3.1,  1.475,  3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23,  2.0,  1.57,  1.5]]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Thread lock for keyboard safety
lock = threading.Lock()

# ------------------------------
# Key Bindings
# ------------------------------
key_to_joint_increase = {'w': 0, 'a': 1, 'r': 2, 'q': 3, 'g': 4, 'z': 5}
key_to_joint_decrease = {'s': 0, 'd': 1, 'f': 2, 'e': 3, 't': 4, 'c': 5}
keys_pressed = {}

# ------------------------------
# Keyboard Listener
# ------------------------------
def on_press(key):
    try:
        k = key.char.lower()
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()
                target_gpos = init_gpos.copy()
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ------------------------------
# Initialize Rerun
# ------------------------------
rr.init("so101_teleop_rerun", spawn=True)

# Load URDF and build joint paths
urdf_path = Path("so101/so101.urdf")
rr.log_file_from_path(str(urdf_path))

# Build joint paths from URDF structure (similar to lerobot's _build_joint_paths)
def build_joint_paths(urdf_path):
    """Build hierarchical joint paths from URDF"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get('name', 'robot')
    
    # Map joint name -> (parent, child, axis)
    joint_info = {}
    for joint in root.findall('.//joint[@type="revolute"]'):
        name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        axis_elem = joint.find('axis')
        axis = [float(x) for x in axis_elem.get('xyz').split()] if axis_elem is not None else [0, 0, 1]
        joint_info[name] = {'parent': parent, 'child': child, 'axis': axis}
    
    # Build hierarchical paths: parent/joint/child
    child_to_joint = {info['child']: (name, info) for name, info in joint_info.items()}
    cache = {}
    
    def build_path(child):
        if child in cache:
            return cache[child]
        if child not in child_to_joint:
            # Base link
            path = child
        else:
            joint_name, info = child_to_joint[child]
            parent_path = build_path(info['parent'])
            path = f"{parent_path}/{joint_name}/{child}"
        cache[child] = path
        return path
    
    joint_paths = {}
    for name, info in joint_info.items():
        joint_paths[name] = {
            'path': build_path(info['child']),
            'axis': info['axis']
        }
    
    return robot_name, joint_paths

robot_name, joint_paths = build_joint_paths(urdf_path)

# ------------------------------
# Main Loop
# ------------------------------
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

x_new, y_new = init_gpos[0], 0.0

try:
    start = time.time()
    while time.time() - start < 1000:
        step_start = time.time()

        # ====== Handle keyboard input ======
        with lock:
            for k, direction in keys_pressed.items():
                if k in ['w', 's', 'a', 'd']:
                    if k == 'w':
                        move_x, move_y = POSITION_INCREMENT, 0
                    elif k == 's':
                        move_x, move_y = -POSITION_INCREMENT, 0
                    elif k == 'a':
                        move_x, move_y = 0, POSITION_INCREMENT
                    elif k == 'd':
                        move_x, move_y = 0, -POSITION_INCREMENT

                    angle_curr = mjdata.qpos[qpos_indices][0]
                    forward_curr = target_gpos_last[0]
                    x_curr, y_curr = forward_curr * np.cos(angle_curr), forward_curr * np.sin(angle_curr)
                    x_new = x_curr + move_x
                    y_new = y_curr + move_y

                    theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
                    forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)

                    target_qpos[0] += theta_update
                    target_gpos[0] += forward_update

                    target_qpos[0] = np.clip(target_qpos[0], control_qlimit[0][0], control_qlimit[1][0])
                    target_gpos[0] = np.clip(target_gpos[0], control_glimit[0][0], control_glimit[1][0])

                elif k in ['r', 'f']:
                    move_dir = 1 if k == 'r' else -1
                    target_gpos[2] += move_dir * POSITION_INCREMENT
                    target_gpos[2] = np.clip(target_gpos[2], control_glimit[0][2], control_glimit[1][2])

                elif k in ['q', 'e', 't', 'g']:
                    ori_idx = {'q': 3, 'e': 3, 't': 4, 'g': 4}[k]
                    move_dir = 1 if k in ['q', 'g'] else -1
                    target_gpos[ori_idx] += move_dir * POSITION_INCREMENT * 4
                    target_gpos[ori_idx] = np.clip(
                        target_gpos[ori_idx],
                        control_glimit[0][ori_idx],
                        control_glimit[1][ori_idx],
                    )

                elif k in ['z', 'c']:
                    move_dir = 1 if k == 'z' else -1
                    target_gpos[5] += move_dir * POSITION_INCREMENT
                    target_gpos[5] = np.clip(target_gpos[5], control_glimit[0][5], control_glimit[1][5])

        # ====== IK, Manipulability ======
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
        jacobian = return_jacobian(fd_qpos, robot=robot)
        m_value, condition = manipulability(jacobian)

        # ====== Apply motion ======
        if ik_success:
            target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
            mjdata.qpos[qpos_indices] = target_qpos
            mujoco.mj_step(mjmodel, mjdata)
            target_gpos_last = target_gpos.copy()
        else:
            target_gpos = target_gpos_last.copy()

        # ====== Rerun Visualization ======
        end_effector_pos = [y_new, -x_new, target_gpos[2]]
        rr.log("end_effector", rr.Transform3D(translation=end_effector_pos))
        rr.log("end_effector/point", rr.Points3D([end_effector_pos], radii=0.01, colors=[255, 0, 0]))
        rr.log("manipulability/value", rr.Scalars(m_value))
        rr.log("manipulability/condition", rr.Scalars(condition))

        # Log joint angles for URDF visualization (matching lerobot's approach)
        # New URDF joint names matching MuJoCo XML
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        for i, joint_name in enumerate(joint_names):
            if joint_name not in joint_paths:
                continue
            joint_angle = target_qpos[i]
            joint_data = joint_paths[joint_name]
            path = f"{robot_name}/{joint_data['path']}"
            axis = joint_data['axis']
            rotation = RotationAxisAngle(axis=axis, angle=joint_angle)
            rr.log(path, rr.Transform3D(rotation=rotation))

        rr.log("joint_angles", rr.Scalars(target_qpos[:6]))

        # Maintain simulation timing
        time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()
