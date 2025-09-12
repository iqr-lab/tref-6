import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_gripper_pose(eef_pos, eef_euler_deg, offset_z=0.03):
    rot = R.from_euler('xyz', eef_euler_deg, degrees=True)
    rot_matrix = rot.as_matrix()

    T_base_to_eef = np.eye(4)
    T_base_to_eef[:3, :3] = rot_matrix
    T_base_to_eef[:3, 3] = eef_pos

    offset_vec = np.array([0, 0, offset_z, 1.0])
    gripper_pos_homog = T_base_to_eef @ offset_vec
    gripper_pos = gripper_pos_homog[:3]
    gripper_euler = eef_euler_deg  # same orientation

    return gripper_pos, gripper_euler

def main():
    input_file = "../demonstrations/door_open/trajectory.json"
    output_file = "../demonstrations/door_open/gripper_trajectory.json"

    with open(input_file, "r") as f:
        traj = json.load(f)

    gripper_traj = []
    for point in traj:
        eef_pos = np.array([point["x"], point["y"], point["z"]])
        eef_euler_deg = [point["theta_x"], point["theta_y"], point["theta_z"]]
        gripper_pos, gripper_euler = compute_gripper_pose(eef_pos, eef_euler_deg)
        
        gripper_point = {
            "x": gripper_pos[0],
            "y": gripper_pos[1],
            "z": gripper_pos[2],
            "theta_x": gripper_euler[0],
            "theta_y": gripper_euler[1],
            "theta_z": gripper_euler[2],
            "gripper": point["gripper"]
        }
        gripper_traj.append(gripper_point)

    # Save output with the same content
    with open(output_file, "w") as f:
        json.dump(gripper_traj, f, indent=2)

if __name__ == "__main__":
    main()
