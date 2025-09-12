import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


# --- Load files ---
with open("door_open.json", "r") as f:
    traj = json.load(f)

with open("door_open_local_frame_in_base.json", "r") as f:
    frame_open = json.load(f)

with open("door_45_inverse_local_frame_in_base.json", "r") as f:
    frame_down = json.load(f)

# --- Build transformation matrices ---
def build_transform(origin, x_axis, y_axis, z_axis):
    R_mat = np.array([x_axis, y_axis, z_axis]).T  # column-wise
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = origin
    return T

T_open = build_transform(frame_open["origin"], frame_open["x_axis"], frame_open["y_axis"], frame_open["z_axis"])
T_down = build_transform(frame_down["origin"], frame_down["x_axis"], frame_down["y_axis"], frame_down["z_axis"])
T_base_to_down = T_down @ np.linalg.inv(T_open)

# --- Filter second phase ---
second_phase = [pt for pt in traj if pt["gripper"] > 0.5]

# --- Transform points ---
def transform_point_corrected(p, T):
    p_hom = np.array([p["x"], p["y"], p["z"], 1.0])
    p_new = T @ p_hom
    return p_new[:3]

original_xyz = np.array([[pt["x"], pt["y"], pt["z"]] for pt in second_phase])
transformed_xyz = np.array([transform_point_corrected(pt, T_base_to_down) for pt in second_phase])


# Convert Euler angles to rotation matrices
rotations_base = [R.from_euler("xyz", [pt["theta_x"], pt["theta_y"], pt["theta_z"]], degrees=True) for pt in second_phase]

# Step 1: Get R_0 (start) in base frame
R_0 = rotations_base[0]

# Step 2: Get R_open_to_down = R_down @ R_open^{-1}
R_open = T_open[:3, :3]
R_down = T_down[:3, :3]
R_open_to_down = R_down @ np.linalg.inv(R_open)

# Step 3: Transform R_0 to door_down frame
rotations_transformed = [
    R.from_matrix(R_open_to_down @ R_t.as_matrix()) for R_t in rotations_base
]

# Step 5: Save everything
transformed_traj_with_rot = []
for pt, new_xyz, R_new in zip(second_phase, transformed_xyz, rotations_transformed):
    theta_x, theta_y, theta_z = R_new.as_euler("xyz", degrees=True)
    transformed_traj_with_rot.append({
        "x": float(new_xyz[0]),
        "y": float(new_xyz[1]),
        "z": float(new_xyz[2]),
        "theta_x": float(theta_x),
        "theta_y": float(theta_y),
        "theta_z": float(theta_z),
        "gripper": pt["gripper"]
    })

with open("transformed_second_phase_with_rotation.json", "w") as f:
    json.dump(transformed_traj_with_rot, f, indent=2)

print("Saved transformed trajectory with rotation to transformed_second_phase_with_rotation.json")