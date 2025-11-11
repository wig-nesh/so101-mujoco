import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot import lerobot_IK, lerobot_FK, create_so101
import threading

np.set_printoptions(linewidth=200)

# Set up the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# Create robot
robot = create_so101()

# Define joint limits
control_qlimit = [[-2.1, -3.1, -0.0, -1.375,  -1.57, -0.15],
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Circular path parameters
CIRCLE_ORIGIN = np.array([0.0, -0.25, 0.08])  # Center of the circle in 3D space (x, y, z)
CIRCLE_RADIUS = 0.05  # Radius of the circle
CIRCLE_PLANE = 'xz'  # Plane: 'xy', 'xz', or 'yz'
CIRCLE_SPEED = 0.01  # Angular speed (radians per step)

# Continuous angle for smooth motion
current_angle = 0.0

# Thread-safe lock
lock = threading.Lock()

try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:

        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()

            with lock:
                # Compute target position based on current angle
                if CIRCLE_PLANE == 'xy':
                    x = CIRCLE_ORIGIN[0] + CIRCLE_RADIUS * np.cos(current_angle)
                    y = CIRCLE_ORIGIN[1] + CIRCLE_RADIUS * np.sin(current_angle)
                    z = CIRCLE_ORIGIN[2]
                elif CIRCLE_PLANE == 'xz':
                    x = CIRCLE_ORIGIN[0] + CIRCLE_RADIUS * np.cos(current_angle)
                    y = CIRCLE_ORIGIN[1]
                    z = CIRCLE_ORIGIN[2] + CIRCLE_RADIUS * np.sin(current_angle)
                elif CIRCLE_PLANE == 'yz':
                    x = CIRCLE_ORIGIN[0]
                    y = CIRCLE_ORIGIN[1] + CIRCLE_RADIUS * np.cos(current_angle)
                    z = CIRCLE_ORIGIN[2] + CIRCLE_RADIUS * np.sin(current_angle)
                else:
                    raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")
                
                target_pos = np.array([x, y, z])
                target_qpos[0] = np.arctan2(target_pos[0], -target_pos[1])
                target_gpos[0] = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
                target_gpos[2] = target_pos[2]

                # Increment angle
                current_angle += CIRCLE_SPEED

            # Compute IK
            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

            if ik_success:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos

                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
            else:
                print("IK failed for angle:", current_angle)

            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    viewer.close()