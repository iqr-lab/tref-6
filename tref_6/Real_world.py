import json
import numpy as np
from force_point_fitter_3d import ForcePointFitter3D

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import cv2
from scipy.spatial.transform import Rotation as R



def project_point_to_image(p_final: np.ndarray):
    """
    Projects a 3D point (p_final) into the 2D image space using known camera intrinsics and extrinsics,
    and visualizes it on the RealSense image.
    
    Args:
        p_final (np.ndarray): The 3D point in robot frame, shape (3,)
    """
    # --- Camera intrinsics ---
    K = np.array([
        [386.154541015625, 0, 323.2745666503906],
        [0, 385.5382385253906, 235.47900390625],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.05744517, 0.06809367, 0.00035157, 0.00067837, -0.02224411])

    T_cam_to_base = np.array([-0.304, 0.350, 0.28], dtype=np.float64)
    R_cam_nominal = np.array([
        [0,  0,  1],
        [-1, 0,  0],
        [0, -1,  0]
    ], dtype=np.float64)

    # --- Invert to get Base → Camera ---
    R_z = R.from_euler('z', -10, degrees=True).as_matrix()
    R_cam_to_base = R_z @ R_cam_nominal

    R_base_to_cam = R_cam_to_base.T
    T_base_to_cam = -R_base_to_cam @ T_cam_to_base

    # --- Build transformation matrix ---
    rvec, _ = cv2.Rodrigues(R_base_to_cam)
    tvec = T_base_to_cam.reshape(3, 1)

    # --- Load image---
    img_path = '../demonstrations/door_open/image.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Project point ---
    point_3d = p_final.reshape(1, 1, 3)
    projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist_coeffs)
    u, v = projected[0, 0]

    # --- Visualize ---
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.scatter(u, v, color='red', s=80, label='Projected p_final')
    plt.title('Projected Inferred Point (p_final) onto Image')
    plt.legend()
    plt.axis('off')
    plt.show()

def resample_equal_distance(x):
    num_points = len(x)

    # Compute arc length (cumulative distance)
    distances = np.linalg.norm(np.diff(x, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]

    # Interpolate each coordinate as a function of arc length
    interpolators = [
        interp1d(cumulative_dist, x[:, i], kind='linear') for i in range(3)
    ]

    # Target equally spaced distances
    target_distances = np.linspace(0, total_length, num_points)

    # Resample each coordinate
    x_resampled = np.stack([
        interpolator(target_distances) for interpolator in interpolators
    ], axis=-1)

    return x_resampled

def curvature_based_dt(x, k=1.0, min_dt=0.01, max_dt=0.1, epsilon=1e-4):
    d = np.diff(x, axis=0)  # (T-1, 3)
    d1 = d[:-1]
    d2 = d[1:]

    # Normalize directions
    d1_norm = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + epsilon)
    d2_norm = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + epsilon)

    # Compute angle between vectors
    dot = np.sum(d1_norm * d2_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot)  # in radians

    mean_curvature = np.mean(angles)
    
    # Use positive correlation: more curvature → larger dt
    dt = k * (mean_curvature + epsilon)
    print(dt)
    return float(np.clip(dt, min_dt, max_dt))

def visualize_run_two_points(x, a, result):

    p1, p2 = result['p1'], result['p2']
    path1, path2 = result['path1'], result['path2']
    switch_step = result['switch_step']


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', linewidth=2)
    ax.scatter(*x[0], c='blue', s=100, label='Start')
    ax.scatter(*x[-1], c='purple', s=100, label='End')


    # Inferred points
    ax.scatter(*p1, c='red', s=100, label='Inferred A')
    ax.scatter(*p2, c='green', s=100, label='Inferred B')

    # Optimization paths
    ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], 'r--', linewidth=1, label='Opt Path A')
    ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], 'g--', linewidth=1, label='Opt Path B')

    # Switch point
    ax.scatter(*x[switch_step], c='black', s=80, label='Switch Point', marker='X')
    
    #ax.scatter(*x[50], c='gray', s=80, marker='X', label='True Switch')

    ax.set_title(f'Fitted Point')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_run_one_points(x, p_final, p_path, switch_step, repreduced_traj=None, p_new = None):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', linewidth=2)
    ax.scatter(*x[0], c='blue', s=100, label='Start')
    ax.scatter(*x[-1], c='purple', s=100, label='End')


    # Inferred points
    ax.scatter(*p_final, c='red', s=100, label='Inferred Point')

    # Optimization paths
    ax.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], 'r--', linewidth=1, label='Opt Path')

    # Switch point
    ax.scatter(*x[switch_step], c='black', s=80, label='Switch Point', marker='X')
    
    if repreduced_traj is not None:
        ax.plot(repreduced_traj[:, 0], repreduced_traj[:, 1], repreduced_traj[:, 2],
                '--', linewidth=2, label='Reproduced Trajectory')
    
    if p_new is not None:
        ax.scatter(*p_new, c='red', s=100, label='New Point')

    ax.set_title(f'Demonstration Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def generalize_trajectory_via_influence(
    traj_demo: np.ndarray,  # (N, 3) original trajectory
    start_demo: np.ndarray, 
    goal_demo: np.ndarray, 
    influence_demo: np.ndarray, 
    start_new: np.ndarray, 
    goal_new: np.ndarray, 
    influence_new: np.ndarray
) -> np.ndarray:
    """
    Generalizes a trajectory from demo to a new scene using scaling based on:
    - Distance between start and goal
    - Distance from influence point to the start-goal line

    Returns:
        traj_new: (N, 3) transformed trajectory in new scene
    """

    # --- Demo Frame ---
    d_demo = goal_demo - start_demo
    d_demo_norm = d_demo / np.linalg.norm(d_demo)
    
    # Project trajectory to demo frame
    traj_rel = traj_demo - start_demo
    alpha = traj_rel @ d_demo_norm  # projection along start-goal
    proj_on_line = np.outer(alpha, d_demo_norm)
    deviation = traj_rel - proj_on_line  # orthogonal component
    beta = np.linalg.norm(deviation, axis=1)
    beta_sign = np.sign(np.sum(deviation @ (influence_demo - start_demo)))  # +1 or -1
    beta = beta * beta_sign

    # Distance metrics
    dist_goal_demo = np.linalg.norm(goal_demo - start_demo)
    dist_inf_demo = np.linalg.norm(np.cross(influence_demo - start_demo, d_demo)) / np.linalg.norm(d_demo)

    # --- New Frame ---
    d_new = goal_new - start_new
    d_new_norm = d_new / np.linalg.norm(d_new)
    dist_goal_new = np.linalg.norm(goal_new - start_new)
    dist_inf_new = np.linalg.norm(np.cross(influence_new - start_new, d_new)) / np.linalg.norm(d_new)

    # Scaling factors
    scale_d = dist_goal_new / dist_goal_demo
    scale_n = dist_inf_new / dist_inf_demo if dist_inf_demo > 1e-6 else 0.0

    v_inf_new = influence_new - start_new
    n_new = v_inf_new - np.dot(v_inf_new, d_new_norm) * d_new_norm
    if np.linalg.norm(n_new) < 1e-6:
        n_new = np.random.randn(3)
    n_new /= np.linalg.norm(n_new)

    # Reconstruct new trajectory
    traj_new = (
        start_new[None, :] +
        alpha[:, None] * scale_d * d_new_norm[None, :] +
        beta[:, None] * scale_n * n_new[None, :]
    )

    return traj_new

# --- Load and preprocess the trajectory---
with open('../demonstrations/door_open/trajectory.json', 'r') as f:
    data = json.load(f)

# Extract position and gripper state
#x = np.array([[p['x'], p['y'], p['z']] for p in data])
x_raw = np.array([[p['x'], p['y'], p['z']] for p in data])
# x = resample_equal_distance(x_raw)
gripper = np.array([p['gripper'] for p in data])

# Find the index where gripper first becomes >= 0.5
split_idx = np.argmax(gripper >= 0.5)

# Split the trajectory at the first occurrence
x_phase1 = x_raw[:split_idx]
x_phase2 = x_raw[split_idx:]

n_phase1 = len(x_phase1)
n_phase2 = len(x_phase2)

x_phase1_resampled = resample_equal_distance(x_phase1)
x_phase1_resampled = gaussian_filter1d(x_phase1_resampled, sigma=2, axis=0)
x_phase2_resampled = resample_equal_distance(x_phase2)
x_phase2_resampled = gaussian_filter1d(x_phase2_resampled, sigma=2, axis=0)

x = np.vstack([x_phase1_resampled, x_phase2_resampled])
dt = 0.1
v = np.gradient(x_phase2_resampled, dt, axis=0)
a = np.gradient(v, dt, axis=0)

# Known switch point based on gripper change
switch_step = np.argmax(gripper == 1)
switch_ratio = switch_step / len(x)

# fitter = ForcePointFitter3D(x, a)
# result = fitter.fit_two_points_best_switch(method='force_residual', switch=switch_ratio)
# --- Visualize ---
#visualize_run_two_points(x, a, result)

fitter = ForcePointFitter3D(x_phase2_resampled, a)
p_final, _, p_path, _ = fitter.fit_one_point(method='force_residual')


traj_demo = x_phase2_resampled
start_demo = traj_demo[0]
goal_demo = traj_demo[-1]
influence_demo = p_final  # inferred from trajectory

# Define new scenario (you can change these values)
start_new = start_demo
goal_new = np.array([0.7, -0.15, 0.15])
influence_new =np.array([0.6, 0.1, 0.175])  # assume object moved

# Warp trajectory
traj_new = generalize_trajectory_via_influence(
    traj_demo,
    start_demo, goal_demo, influence_demo,
    start_new, goal_new, influence_new
)

#visualize_run_one_points(x, p_final, p_path, switch_step, traj_new, influence_new)
project_point_to_image(p_final)


