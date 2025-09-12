import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load local frame and handle ---
with open("../features/door_open/local_frame.json", "r") as f:
    frame_data = json.load(f)

origin = np.array(frame_data["origin"])
x_axis = np.array(frame_data["x_axis"])
y_axis = np.array(frame_data["y_axis"])
z_axis = np.array(frame_data["z_axis"])
handle = np.array(frame_data["handle_position"])

# --- Load gripper trajectory ---
with open("../demonstrations/door_open/gripper_trajectory.json", "r") as f:
    gripper_traj = json.load(f)

gripper_positions = np.array([[p["x"], p["y"], p["z"]] for p in gripper_traj])

# --- Load original EEF trajectory ---
with open("../demonstrations/door_open/trajectory.json", "r") as f:
    eef_traj = json.load(f)

eef_positions = np.array([[p["x"], p["y"], p["z"]] for p in eef_traj])

# --- Plotting ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot gripper trajectory
ax.plot(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
        label="Gripper Trajectory", color='orange', linewidth=2)

# Plot EEF trajectory
ax.plot(eef_positions[:, 0], eef_positions[:, 1], eef_positions[:, 2],
        label="EEF Trajectory", color='purple', linestyle='--', linewidth=1.5)

# Plot local frame axes
scale = 0.05
ax.quiver(*origin, *(x_axis * scale), color='r', label='x_axis')
ax.quiver(*origin, *(y_axis * scale), color='g', label='y_axis')
ax.quiver(*origin, *(z_axis * scale), color='b', label='z_axis')

# Plot handle position
ax.scatter(*handle, color='k', s=50, label='Handle')

# Formatting
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Gripper vs EEF Trajectory with Local Frame and Handle")
ax.legend()
plt.tight_layout()
plt.savefig("gripper_vs_eef_trajectory.png", dpi=300)
