import numpy as np
import json
from pydmps.dmp_discrete import DMPs_discrete
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

def compute_gripper_pose(eef_pos, eef_euler_deg, offset_z=0.03):
    rot = R.from_euler('xyz', eef_euler_deg, degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = eef_pos
    gripper_pos = (T @ np.array([0, 0, offset_z, 1.0]))[:3]
    return gripper_pos, eef_euler_deg

def compute_eef_pose(gripper_pos, gripper_euler_deg, offset_z=0.03):
    rot = R.from_euler('xyz', gripper_euler_deg, degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = gripper_pos
    eef_pos = (T @ np.array([0, 0, -offset_z, 1.0]))[:3]
    return eef_pos, gripper_euler_deg

# === Load and convert raw trajectory to gripper-space ===
with open("../demonstrations/door_open/trajectory.json", "r") as f:
    raw_traj_data = json.load(f)

traj_data = []
for point in raw_traj_data:
    eef_pos = np.array([point["x"], point["y"], point["z"]])
    eef_rot = [point["theta_x"], point["theta_y"], point["theta_z"]]
    gripper_pos, gripper_rot = compute_gripper_pose(eef_pos, eef_rot)
    traj_data.append({
        "x": float(gripper_pos[0]),
        "y": float(gripper_pos[1]),
        "z": float(gripper_pos[2]),
        "theta_x": float(gripper_rot[0]),
        "theta_y": float(gripper_rot[1]),
        "theta_z": float(gripper_rot[2]),
        "gripper": point["gripper"]
    })

# === Load handle positions ===
with open("../features/door_open/local_frame.json", "r") as f:
    old_frame = json.load(f)
with open("../generalized/failure/door_up_135/local_frame.json", "r") as f:
    new_frame = json.load(f)

old_handle = np.array(old_frame["handle_position"])
new_handle = np.array(new_frame["handle_position"])

# === Split phase 1 and 2 ===
phase1_traj = []
second_phase_traj = []
triggered = False
for p in traj_data:
    if not triggered:
        phase1_traj.append(p)
    if p["gripper"] > 0.5:
        triggered = True
    if triggered:
        second_phase_traj.append(p)

# === Train and rollout DMP for Phase 1 (to new handle) ===
phase1_pos = np.array([[p["x"], p["y"], p["z"]] for p in phase1_traj])
phase1_quat = np.array([
    R.from_euler("xyz", [p["theta_x"], p["theta_y"], p["theta_z"]], degrees=True).as_quat()
    for p in phase1_traj
])

dmp1_x = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_y = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_z = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_qx = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_qy = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_qz = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp1_qw = DMPs_discrete(n_dmps=1, n_bfs=50)

dmp1_x.imitate_path(phase1_pos[:, 0])
dmp1_y.imitate_path(phase1_pos[:, 1])
dmp1_z.imitate_path(phase1_pos[:, 2])
dmp1_qx.imitate_path(phase1_quat[:, 0])
dmp1_qy.imitate_path(phase1_quat[:, 1])
dmp1_qz.imitate_path(phase1_quat[:, 2])
dmp1_qw.imitate_path(phase1_quat[:, 3])

dmp1_x.goal = np.array([new_handle[0]])
dmp1_y.goal = np.array([new_handle[1]])
dmp1_z.goal = np.array([new_handle[2]])

px, _, _ = dmp1_x.rollout()
py, _, _ = dmp1_y.rollout()
pz, _, _ = dmp1_z.rollout()
qx, _, _ = dmp1_qx.rollout()
qy, _, _ = dmp1_qy.rollout()
qz, _, _ = dmp1_qz.rollout()
qw, _, _ = dmp1_qw.rollout()

pos1_gen = np.stack([px.squeeze(), py.squeeze(), pz.squeeze()], axis=1)
quat1_gen = np.stack([qx.squeeze(), qy.squeeze(), qz.squeeze(), qw.squeeze()], axis=1)
quat1_gen /= np.linalg.norm(quat1_gen, axis=1, keepdims=True)

reaching_traj = []
for i, (pos, quat) in enumerate(zip(pos1_gen, quat1_gen)):
    euler = R.from_quat(quat).as_euler("xyz", degrees=True)
    eef_pos, eef_rot = compute_eef_pose(pos, euler)
    reaching_traj.append({
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "theta_x": float(eef_rot[0]),
        "theta_y": float(eef_rot[1]),
        "theta_z": float(eef_rot[2]),
        "gripper": float(phase1_traj[i]["gripper"])
    })

with open("../generalized/failure/door_up_135/baseline_reaching_pose.json", "w") as f:
    json.dump(reaching_traj, f, indent=2)
print("Saved reaching trajectory")

# === Train and rollout DMP for Phase 2 (goal = handle + delta_goal) ===
traj2_pos = np.array([[p["x"], p["y"], p["z"]] for p in second_phase_traj])
traj2_quat = np.array([
    R.from_euler("xyz", [p["theta_x"], p["theta_y"], p["theta_z"]], degrees=True).as_quat()
    for p in second_phase_traj
])
delta_goal = traj2_pos[-1] - old_handle
new_goal = new_handle + delta_goal

dmp2_x = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_y = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_z = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_qx = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_qy = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_qz = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp2_qw = DMPs_discrete(n_dmps=1, n_bfs=50)

dmp2_x.imitate_path(traj2_pos[:, 0])
dmp2_y.imitate_path(traj2_pos[:, 1])
dmp2_z.imitate_path(traj2_pos[:, 2])
dmp2_qx.imitate_path(traj2_quat[:, 0])
dmp2_qy.imitate_path(traj2_quat[:, 1])
dmp2_qz.imitate_path(traj2_quat[:, 2])
dmp2_qw.imitate_path(traj2_quat[:, 3])

dmp2_x.y0 = np.array([new_handle[0]])
dmp2_y.y0 = np.array([new_handle[1]])
dmp2_z.y0 = np.array([new_handle[2]])
dmp2_x.goal = np.array([new_goal[0]])
dmp2_y.goal = np.array([new_goal[1]])
dmp2_z.goal = np.array([new_goal[2]])

px, _, _ = dmp2_x.rollout()
py, _, _ = dmp2_y.rollout()
pz, _, _ = dmp2_z.rollout()
qx, _, _ = dmp2_qx.rollout()
qy, _, _ = dmp2_qy.rollout()
qz, _, _ = dmp2_qz.rollout()
qw, _, _ = dmp2_qw.rollout()

pos2_gen = np.stack([px.squeeze(), py.squeeze(), pz.squeeze()], axis=1)
quat2_gen = np.stack([qx.squeeze(), qy.squeeze(), qz.squeeze(), qw.squeeze()], axis=1)
quat2_gen /= np.linalg.norm(quat2_gen, axis=1, keepdims=True)

gripper_vals = np.array([p["gripper"] for p in second_phase_traj])
f_interp = interp1d(np.linspace(0, 1, len(gripper_vals)), gripper_vals)
gripper_resampled = f_interp(np.linspace(0, 1, len(pos2_gen)))

goal_traj = []
for i, (pos, quat) in enumerate(zip(pos2_gen, quat2_gen)):
    euler = R.from_quat(quat).as_euler("xyz", degrees=True)
    eef_pos, eef_rot = compute_eef_pose(pos, euler)
    goal_traj.append({
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "theta_x": float(eef_rot[0]),
        "theta_y": float(eef_rot[1]),
        "theta_z": float(eef_rot[2]),
        "gripper": float(gripper_resampled[i])
    })

with open("../generalized/failure/door_up_135/baseline_goal_from_handle.json", "w") as f:
    json.dump(goal_traj, f, indent=2)
print("Saved goal trajectory")

# === Load original frame data for plotting ===
origin = np.array(old_frame["origin"])
x_axis = np.array(old_frame["x_axis"])
y_axis = np.array(old_frame["y_axis"])
z_axis = np.array(old_frame["z_axis"])

new_origin = np.array(new_frame["origin"])
new_x_axis = np.array(new_frame["x_axis"])
new_y_axis = np.array(new_frame["y_axis"])
new_z_axis = np.array(new_frame["z_axis"])

# === Re-extract original and generated XYZ ===
original_xyz = np.array([
    [p["x"], p["y"], p["z"]]
    for p in traj_data if p["gripper"] > 0.5
])
generated_xyz = np.array([
    [p["x"], p["y"], p["z"]]
    for p in goal_traj
])
phase1_pos_gen = np.array([
    [p["x"], p["y"], p["z"]]
    for p in reaching_traj
])



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original and generated trajectories
ax.plot(*original_xyz.T, label="Original Trajectory", color='gray')
ax.plot(*generated_xyz.T, label="Generated Trajectory", color='red')

ax.plot(*phase1_pos.T, label="Phase 1 Original", color='black', linestyle='--')
ax.plot(*phase1_pos_gen.T, label="Phase 1 Generated", color='orange', linestyle='--')

# Plot handle positions
ax.scatter(*old_handle, color='blue', label='Original Handle', s=50)
ax.scatter(*new_handle, color='green', label='New Handle', s=50)

# Plot original frame
frame_len = 0.05
ax.quiver(*origin, *x_axis, length=frame_len, color='r', label='X axis (orig)')
ax.quiver(*origin, *y_axis, length=frame_len, color='g', label='Y axis (orig)')
ax.quiver(*origin, *z_axis, length=frame_len, color='b', label='Z axis (orig)')

# Plot new frame (semi-transparent)
ax.quiver(*new_origin, *new_x_axis, length=frame_len, color='r', alpha=0.5)
ax.quiver(*new_origin, *new_y_axis, length=frame_len, color='g', alpha=0.5)
ax.quiver(*new_origin, *new_z_axis, length=frame_len, color='b', alpha=0.5)

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Baseline: Original vs. Generalized Trajectory with Frames")
ax.legend()
plt.tight_layout()
plt.savefig("../generalized/failure/door_up_135/dmp_baseline.png")
print("Saved plot to generalized_trajectory_plot.png")
