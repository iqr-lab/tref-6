import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R


# --- Load 3D Trajectory Data ---
with open('../demonstrations/door_open/trajectory.json', 'r') as f:
    data = json.load(f)
x = np.array([[p['x'], p['y'], p['z']] for p in data])
x_hom = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# --- Load RealSense Image ---
img = cv2.imread('../demonstrations/door_open/image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- RealSense Camera Intrinsics ---
K = np.array([
    [386.154541015625, 0, 323.2745666503906],
    [0, 385.5382385253906, 235.47900390625],
    [0, 0, 1]
])
dist_coeffs = np.array([-0.05744517, 0.06809367, 0.00035157, 0.00067837, -0.02224411])

# --- Known Transformation: Camera → Base ---
T_cam_to_base = np.array([-0.304, 0.350, 0.28], dtype=np.float64)
R_cam_nominal = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0]
], dtype=np.float64)

# --- Invert to get Base → Camera ---
R_z_base = R.from_euler('z', -10, degrees=True).as_matrix()
R_cam_to_base = R_z_base @ R_cam_nominal

R_base_to_cam = R_cam_to_base.T
T_base_to_cam = -R_base_to_cam @ T_cam_to_base

# --- Build transformation matrix ---
T_robot_to_cam = np.eye(4)
T_robot_to_cam[:3, :3] = R_base_to_cam
T_robot_to_cam[:3, 3] = T_base_to_cam

x_cam = (T_robot_to_cam @ x_hom.T).T[:, :3]

# --- Project 3D trajectory to 2D image ---
pts, _ = cv2.projectPoints(x_cam, np.zeros(3), np.zeros(3), K, dist_coeffs)
pts = pts.squeeze()

# --- Plot image and projected trajectory ---
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.plot(pts[:, 0], pts[:, 1], color='cyan', linewidth=3, label='Projected Trajectory')
plt.title('Projected 3D Trajectory onto Image')
plt.legend()
plt.axis('off')
plt.show()