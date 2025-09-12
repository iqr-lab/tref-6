import numpy as np
import json
from pydmps.dmp_discrete import DMPs_discrete
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d


matplotlib.use('Agg')
# --- Load files ---
with open("../demonstrations/wiping/trajectory.json", "r") as f:
    raw_traj_data = json.load(f)
with open("../features/wiping/local_frame.json", "r") as f:
    frame = json.load(f)
# with open("../demonstrations/wiping/trajectory.json", "r") as f:
#     raw_traj_data = json.load(f)
# with open("../features/wiping/local_frame.json", "r") as f:
#     frame = json.load(f)
with open("../generalized/wiping_red_90/local_frame.json", "r") as f:
    new_frame = json.load(f)

# --- Build transform ---
def build_T(origin, x_axis, y_axis, z_axis):
    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = origin
    return T

def remap_rot_around_x_y(euler_deg):
    # Convert Euler to rotation object
    rot_obj = R.from_euler("xyz", euler_deg, degrees=True)
    angle = rot_obj.magnitude()
    axis = rot_obj.as_rotvec()
    if np.linalg.norm(axis) < 1e-6:
        return euler_deg  # No rotation

    unit_axis = axis / np.linalg.norm(axis)

    # Project rotation magnitude onto X and Y axes
    proj_x = np.dot(unit_axis, np.array([1, 0, 0])) * angle
    proj_y = np.dot(unit_axis, np.array([0, 1, 0])) * angle
    proj_z = np.dot(unit_axis, np.array([0, 0, 1])) * angle

    # Remap these projected components (you can modify this logic!)
    def remap_angle_x(angle):
        # Example: reflect to ±135°
        angle = np.rad2deg(angle)
        if -90 <= angle <= 90:
            return np.deg2rad(180 - angle if angle >= 0 else -180 - angle)
        return np.deg2rad(angle)

    def remap_angle_y(angle):
        angle = np.rad2deg(angle)
        if angle < -90:
            return np.deg2rad(-180 - angle)
        elif angle > 90:
            return np.deg2rad(180 - angle)
        return np.deg2rad(angle)

    new_proj_x = remap_angle_x(proj_x)
    new_proj_y = remap_angle_y(proj_y)

    # Rebuild rotation from remapped axis-angle components
    new_rotvec = (
        new_proj_x * np.array([1, 0, 0]) +
        new_proj_y * np.array([0, 1, 0]) +
        proj_z * np.array([0, 0, 1])  # keep Z the same
    )
    return R.from_rotvec(new_rotvec).as_euler("xyz", degrees=True)

    return angle

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

# Convert to gripper trajectory
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

# --- Extract original frame data ---
origin = np.array(frame["origin"])
x_axis = np.array(frame["x_axis"])
y_axis = np.array(frame["y_axis"])
z_axis = np.array(frame["z_axis"])
handle = np.array(frame["handle_position"])
T_world_to_local = np.linalg.inv(build_T(origin, x_axis, y_axis, z_axis))
R_world_to_local = np.stack([x_axis, y_axis, z_axis], axis=1).T

# --- Extract new frame data ---
new_origin = np.array(new_frame["origin"])
new_x_axis = np.array(new_frame["x_axis"])
new_y_axis = np.array(new_frame["y_axis"])
new_z_axis = np.array(new_frame["z_axis"])
new_handle = np.array(new_frame["handle_position"])
T_local_to_world_new = build_T(new_origin, new_x_axis, new_y_axis, new_z_axis)
R_local_to_world_new = np.stack([new_x_axis, new_y_axis, new_z_axis], axis=1)



# --- Extract second phase and transform ---
traj_world = []
quats_local = []
second_phase_traj = []

triggered = False
for p in traj_data:
    if not triggered and  p["gripper"] > 0.3:
        triggered = True
    if triggered:
        traj_world.append([p["x"], p["y"], p["z"]])
        R_world = R.from_euler("xyz", [p["theta_x"], p["theta_y"], p["theta_z"]], degrees=True).as_matrix()
        R_local = R_world_to_local @ R_world
        quat = R.from_matrix(R_local).as_quat() 
        quats_local.append(quat)
        second_phase_traj.append(p)

phase1_traj = []
for p in traj_data:
    phase1_traj.append(p)
    if p["gripper"] > 0.3:
        break

# --- Extract position and orientation ---
phase1_pos = np.array([[p["x"], p["y"], p["z"]] for p in phase1_traj])
phase1_rot = np.array([
    R.from_euler("xyz", [p["theta_x"], p["theta_y"], p["theta_z"]], degrees=True).as_quat()
    for p in phase1_traj
])

# --- Train DMPs for position ---
dmp_phase1_x = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_phase1_y = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_phase1_z = DMPs_discrete(n_dmps=1, n_bfs=50)

dmp_phase1_x.imitate_path(phase1_pos[:, 0])
dmp_phase1_y.imitate_path(phase1_pos[:, 1])
dmp_phase1_z.imitate_path(phase1_pos[:, 2])

# --- Train DMPs for quaternion ---
dmp_phase1_qx = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_phase1_qy = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_phase1_qz = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_phase1_qw = DMPs_discrete(n_dmps=1, n_bfs=50)

dmp_phase1_qx.imitate_path(phase1_rot[:, 0])
dmp_phase1_qy.imitate_path(phase1_rot[:, 1])
dmp_phase1_qz.imitate_path(phase1_rot[:, 2])
dmp_phase1_qw.imitate_path(phase1_rot[:, 3])



traj_world = np.array(traj_world)
quats_local = np.array(quats_local)


# --- Train DMPs for quaternion (x, y, z, w) ---
dmp_qx = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_qy = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_qz = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_qw = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_qx.imitate_path(quats_local[:, 0])
dmp_qy.imitate_path(quats_local[:, 1])
dmp_qz.imitate_path(quats_local[:, 2])
dmp_qw.imitate_path(quats_local[:, 3])


# --- Transform original trajectory to local frame ---
traj_local = (T_world_to_local @ np.hstack([traj_world, np.ones((len(traj_world), 1))]).T).T[:, :3]
handle_local = (T_world_to_local @ np.append(handle, 1))[:3]
handle_local_new = (np.linalg.inv(T_local_to_world_new) @ np.append(new_handle, 1))[:3]

# --- Compute goal position in new local frame ---
p_end_local = traj_local[-1]
relative_to_origin = p_end_local  # since origin = (0,0,0) in local frame
goal_new_local = relative_to_origin

# --- Train DMP on trajectory relative to original handle ---
#traj_rel = traj_local - handle_local
delta_start = traj_local[0] - handle_local
traj_rel = traj_local - traj_local[0]
dmp_x = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_y = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_z = DMPs_discrete(n_dmps=1, n_bfs=50)
dmp_x.imitate_path(traj_rel[:, 0])
dmp_y.imitate_path(traj_rel[:, 1])
dmp_z.imitate_path(traj_rel[:, 2])

# --- Set new start and goal for DMP ---
p_start_new = handle_local_new + delta_start
p_goal_new = goal_new_local  # same relative to frame origin

print("delta_start:", delta_start)
print("p_goal_new:", p_goal_new)
print("p_start_new:", p_start_new)


dmp_x.y0 = np.array([0.0])
dmp_y.y0 = np.array([0.0])
dmp_z.y0 = np.array([0.0])
dmp_x.goal = np.array([p_goal_new[0] - p_start_new[0]])
dmp_y.goal = np.array([p_goal_new[1] - p_start_new[1]])
dmp_z.goal = np.array([p_goal_new[2] - p_start_new[2]])
# dmp_x.goal = np.array([traj_rel[-1, 0]])
# dmp_y.goal = np.array([traj_rel[-1, 1]])
# dmp_z.goal = np.array([traj_rel[-1, 2]])


# --- Rollout and shift to new handle ---
y_x, _, _ = dmp_x.rollout()
y_y, _, _ = dmp_y.rollout()
y_z, _, _ = dmp_z.rollout()
y_local = np.stack([y_x.squeeze(), y_y.squeeze(), y_z.squeeze()], axis=1)
y_local_shifted = y_local + p_start_new
y_world = (T_local_to_world_new @ np.hstack([y_local_shifted, np.ones((len(y_local_shifted), 1))]).T).T[:, :3]

# --- Rollout quaternion and convert to Euler in world frame ---
qx, _, _ = dmp_qx.rollout()
qy, _, _ = dmp_qy.rollout()
qz, _, _ = dmp_qz.rollout()
qw, _, _ = dmp_qw.rollout()
quats_gen = np.stack([qx.squeeze(), qy.squeeze(), qz.squeeze(), qw.squeeze()], axis=1)

# Normalize quaternions and transform to world frame
theta_world_gen = []


for q in quats_gen:
    q_normalized = q / np.linalg.norm(q)
    R_local = R.from_quat(q_normalized).as_matrix()
    R_world = R_local_to_world_new @ R_local
    euler_world = R.from_matrix(R_world).as_euler("xyz", degrees=True)
    theta_world_gen.append(euler_world)
theta_world_gen = np.array(theta_world_gen)

R_local_to_world_orig = np.stack([x_axis, y_axis, z_axis], axis=1)
R_old_to_new = R_local_to_world_new.T @ R_local_to_world_orig
rel_euler = R.from_matrix(R_old_to_new).as_euler("xyz", degrees=True)


# --- Set new goal for Phase 1 to match the start of generalized trajectory ---
new_goal_pos = y_world[0]
new_goal_quat = R.from_euler("xyz", theta_world_gen[0], degrees=True).as_quat()
new_goal_quat /= np.linalg.norm(new_goal_quat)  # normalize

# --- Update DMP goals for position ---
dmp_phase1_x.goal = np.array([new_goal_pos[0]])
dmp_phase1_y.goal = np.array([new_goal_pos[1]])
dmp_phase1_z.goal = np.array([new_goal_pos[2]])

# --- Update DMP goals for quaternion ---
dmp_phase1_qx.goal = np.array([new_goal_quat[0]])
dmp_phase1_qy.goal = np.array([new_goal_quat[1]])
dmp_phase1_qz.goal = np.array([new_goal_quat[2]])
dmp_phase1_qw.goal = np.array([new_goal_quat[3]])

# --- Rollout Phase 1 ---
px, _, _ = dmp_phase1_x.rollout()
py, _, _ = dmp_phase1_y.rollout()
pz, _, _ = dmp_phase1_z.rollout()
qx, _, _ = dmp_phase1_qx.rollout()
qy, _, _ = dmp_phase1_qy.rollout()
qz, _, _ = dmp_phase1_qz.rollout()
qw, _, _ = dmp_phase1_qw.rollout()

phase1_pos_gen = np.stack([px.squeeze(), py.squeeze(), pz.squeeze()], axis=1)
quats_gen = np.stack([qx.squeeze(), qy.squeeze(), qz.squeeze(), qw.squeeze()], axis=1)

# Normalize and convert to Euler
eulers_gen = [R.from_quat(q / np.linalg.norm(q)).as_euler("xyz", degrees=True) for q in quats_gen]
eulers_gen = np.array(eulers_gen)

# --- Save phase 1 generalized trajectory ---
phase1_generalized = []
for i, (pos, rot) in enumerate(zip(phase1_pos_gen, eulers_gen)):
    original_gripper = phase1_traj[i]["gripper"]

    # Remap angles
    remapped_euler = remap_rot_around_x_y(rot)
    eef_pos, eef_euler = compute_eef_pose(pos, remapped_euler)

    phase1_generalized.append({
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "theta_x": float(eef_euler[0]),
        "theta_y": float(eef_euler[1]),
        "theta_z": float(eef_euler[2]),
        "gripper": float(original_gripper)
    })

with open("../generalized/wiping_red_90/generalized_dmp_reaching_pose.json", "w") as f:
    json.dump(phase1_generalized, f, indent=2)

print("Saved generalized reaching trajectory to generalized_dmp_reaching_pose.json")
# --- Save result ---
generalized_traj = []
# Original gripper values
gripper_vals = np.array([p["gripper"] for p in second_phase_traj])

# Interpolation over [0,1]
f_interp = interp1d(np.linspace(0, 1, len(gripper_vals)), gripper_vals)

# Resample to DMP length (100)
gripper_resampled = f_interp(np.linspace(0, 1, len(y_world)))
for i, (pos, rot) in enumerate(zip(y_world, theta_world_gen)):
    original_gripper = gripper_resampled[i]

    # Remap angles
    remapped_euler = remap_rot_around_x_y(rot)
    eef_pos, eef_euler = compute_eef_pose(pos, remapped_euler)

    generalized_traj.append({
        "x": float(eef_pos[0]),
        "y": float(eef_pos[1]),
        "z": float(eef_pos[2]),
        "theta_x": float(eef_euler[0]),
        "theta_y": float(eef_euler[1]),
        "theta_z": float(eef_euler[2]),
        "gripper": float(original_gripper)
    })

with open("../generalized/wiping_red_90/generalized_dmp_quaternion_pose.json", "w") as f:
    json.dump(generalized_traj, f, indent=2)

print("Saved generalized quaternion-based pose trajectory to generalized_dmp_quaternion_pose.json")

# Re-extract original trajectory from second phase
original_xyz = np.array([
    [p["x"], p["y"], p["z"]]
    for p in traj_data if p["gripper"] > 0.3
])
generated_xyz = np.array([
    [p["x"], p["y"], p["z"]]
    for p in generalized_traj
])

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original and generated trajectories
ax.plot(*original_xyz.T, label="Original Trajectory", color='gray')
ax.plot(*generated_xyz.T, label="Generated Trajectory", color='red')

ax.plot(*phase1_pos.T, label="Phase 1 Original", color='black', linestyle='--')
ax.plot(*phase1_pos_gen.T, label="Phase 1 Generated", color='orange', linestyle='--')

# Plot handle positions
ax.scatter(*handle, color='blue', label='Original Handle', s=50)
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
ax.set_title("Original vs. Generalized Trajectory with Frames")
ax.legend()
plt.tight_layout()
plt.savefig("../generalized/wiping_red_90/generalized_trajectory_plot.png")

