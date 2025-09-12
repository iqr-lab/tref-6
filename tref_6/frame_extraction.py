import numpy as np
import cv2
import json
from pycocotools import mask as mask_utils
from scipy.spatial.transform import Rotation as R

# --- Load data ---
depth_map = np.load("../generalized/red_cube_green_drop/depth.npy")
with open("../generalized/red_cube_green_drop/segmentation.json", "r") as f:
    data = json.load(f)
image = cv2.imread("../generalized/red_cube_green_drop/image.jpg")
seam_mask = mask_utils.decode(data["annotations"][1]["segmentation"]).astype(np.uint8)

# --- Camera intrinsics ---
K = np.array([
    [386.154541015625, 0, 323.2745666503906],
    [0, 385.5382385253906, 235.47900390625],
    [0, 0, 1]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

T_cam_to_base = np.array([-0.304, 0.350, 0.28], dtype=np.float64)
R_cam_nominal = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0]
], dtype=np.float64)

# Rotation from base Z by -10°
R_z_base = R.from_euler('z', -10, degrees=True).as_matrix()
R_cam_to_base = R_z_base @ R_cam_nominal

# --- Pixel coordinates ---
seam_box = data["annotations"][1]["bbox"]
handle_box = data["annotations"][0]["bbox"]

u_seam = int((seam_box[0] + seam_box[2]) / 2)
v_seam = int((seam_box[1] + seam_box[3]) / 2)
u_handle = int((handle_box[0] + handle_box[2]) / 2)
v_handle = int((handle_box[1] + handle_box[3]) / 2)

# --- Pixel to 3D ---
def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    z = depth[v, u] / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

p_seam = pixel_to_3d(u_seam, v_seam, depth_map, fx, fy, cx, cy)
p_handle = pixel_to_3d(u_handle, v_handle, depth_map, fx, fy, cx, cy)

# --- Surface normal at seam ---
def compute_surface_normal(mask, depth, u, v, fx, fy, cx, cy, window=5):
    points = []
    h, w = mask.shape
    for du in range(-window, window+1):
        for dv in range(-window, window+1):
            nu, nv = u + du, v + dv
            if 0 <= nu < w and 0 <= nv < h and mask[nv, nu]:
                pt = pixel_to_3d(nu, nv, depth, fx, fy, cx, cy)
                if np.all(np.isfinite(pt)):
                    points.append(pt)
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered)
    normal = vh[-1]
    #normal = vh[0]
    return normal / np.linalg.norm(normal)

# --- z-axis: surface normal ---
z_axis = compute_surface_normal(seam_mask, depth_map, u_seam, v_seam, fx, fy, cx, cy)
if z_axis[2] > 0:
    z_axis = -z_axis
z_axis /= np.linalg.norm(z_axis)

# --- y-axis: seam → handle direction ---
y_axis = p_handle - p_seam
y_axis /= np.linalg.norm(y_axis)

# --- x = y × z, then re-orthogonalize y = z × x ---
x_axis = np.cross(y_axis, z_axis)
x_axis /= np.linalg.norm(x_axis)

y_axis = np.cross(z_axis, x_axis)
y_axis /= np.linalg.norm(y_axis)


# --- Final 3x3 rotation matrix ---
R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)

# --- Draw axes on image ---
def project_to_image(pt3d, K):
    pt3d = pt3d.reshape(3, 1)
    pt2d = K @ pt3d
    pt2d /= pt2d[2]
    return int(pt2d[0].item()), int(pt2d[1].item())

length = 0.1
axes_3d = [p_seam + length * R_cam[:, i] for i in range(3)]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
labels = ['x', 'y', 'z']

for i, pt3d in enumerate(axes_3d):
    pt2d_start = (u_seam, v_seam)
    pt2d_end = project_to_image(pt3d, K)
    cv2.arrowedLine(image, pt2d_start, pt2d_end, colors[i], thickness=4, tipLength=0.3)
    cv2.putText(image, labels[i], pt2d_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

# Draw origin
cv2.circle(image, (u_seam, v_seam), 6, (0, 255, 255), -1)
cv2.circle(image, (u_seam, v_seam), 10, (0, 0, 0), 2)

# Save and print
cv2.imwrite("../generalized/red_cube_green_drop/image_with_reference_frame.jpg", image)
print("Saved: image_with_reference_frame.jpg")

# --- Print the frame ---
print("=== Local Reference Frame (Camera Coordinates) ===")
print("Origin (seam center):", p_seam)
print("X-axis (orthogonal):", x_axis)
print("Y-axis (to handle):", y_axis)
print("Z-axis (surface normal):", z_axis)

origin_base = R_cam_to_base @ p_seam + T_cam_to_base
p_handle_base = R_cam_to_base @ p_handle + T_cam_to_base
R_base = R_cam_to_base @ R_cam

frame_data = {
    "origin": origin_base.tolist(),
    "x_axis": R_base[:, 0].tolist(),
    "y_axis": R_base[:, 1].tolist(),
    "z_axis": R_base[:, 2].tolist(),
    "handle_position": p_handle_base.tolist()
}

# --- Save to JSON ---
with open("../generalized/red_cube_green_drop/local_frame.json", "w") as f:
    json.dump(frame_data, f, indent=4)

print("\nSaved: local_frame_in_base.json")