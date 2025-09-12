#!/usr/bin/env python3

import sys
import os
import time
import threading
import json
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp


from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION = 20

def reconnect_robot(ip="192.168.2.5", username="admin", password="admin"):
    print("[INFO] Attempting to reconnect to robot...")
    transport = TCPTransport()
    transport.connect(ip, 10000)
    router = RouterClient(transport, RouterClient.basicErrorCallback)
    session_manager = SessionManager(router)
    session_manager.CreateSession(username, password)
    print("[INFO] Reconnection successful.")
    return router, session_manager, BaseClient(router), BaseCyclicClient(router)

def check_for_end_or_abort(e):
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def execute_pose_and_wait(base, pose_dict, index=None):
    action = Base_pb2.Action()
    if index is not None:
        action.name = f"Step_{index}"
    pose = action.reach_pose.target_pose
    pose.x = pose_dict['x']
    pose.y = pose_dict['y']
    pose.z = pose_dict['z']
    pose.theta_x = pose_dict['theta_x']
    pose.theta_y = pose_dict['theta_y']
    pose.theta_z = pose_dict['theta_z']

    e = threading.Event()
    handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())
    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(handle)
    return finished

def example_move_to_home_position(base):
    print("Loading home position from Home.json...")
    home_json_path = os.path.join(os.path.dirname(__file__), "Home.json")
    with open(home_json_path, 'r') as f:
        home_data = json.load(f)

    action = Base_pb2.Action()
    action.name = "Home"
    action.application_data = home_data["jointAnglesGroup"]["jointAngles"][0]["applicationData"]
    action.handle.identifier = home_data["jointAnglesGroup"]["jointAngles"][0]["handle"]["identifier"]
    action.handle.permission = home_data["jointAnglesGroup"]["jointAngles"][0]["handle"]["permission"]
    action.handle.action_type = home_data["jointAnglesGroup"]["jointAngles"][0]["handle"]["actionType"]

    joint_values = home_data["jointAnglesGroup"]["jointAngles"][0]["reachJointAngles"]["jointAngles"]["jointAngles"]
    for j in joint_values:
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = j["jointIdentifier"]
        joint_angle.value = j["value"]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())
    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Home position reached.")
    else:
        print("Timeout while reaching Home position.")
    return finished

def move_and_grab(base, traj_path, gripper_value=0.5):
    print(f"Loading and executing grasp trajectory from {traj_path}...")

    with open(traj_path, 'r') as f:
        trajectory = json.load(f)

    smoothed_traj = interpolate_trajectory(trajectory, num_points=200)

    success = replay_trajectory_smooth(base, smoothed_traj)

    if not success:
        print("Trajectory execution failed before grasp.")
        return False

    print("Grasp complete.")
    return True

def compute_grasp_pose_from_object(object_position):
    offset = np.array([
        0.6333083510398865 - 0.6647581332166377,
        0.1943151354789734 - 0.20050531503754776,
        0.19584530591964722 - 0.18970914764507218
    ])
    grasp_position = (np.array(object_position) + offset).tolist()
    grasp_orientation = [128.24, 0.29, 89.31]
    return grasp_position, grasp_orientation

def interpolate_trajectory(trajectory, num_points=200, angle_threshold_deg=5.0):

    # --- Initial interpolation ---
    positions = np.array([[pt['x'], pt['y'], pt['z']] for pt in trajectory])
    eulers = np.deg2rad([[pt['theta_x'], pt['theta_y'], pt['theta_z']] for pt in trajectory])
    quats = R.from_euler('xyz', eulers).as_quat()
    grippers = np.array([pt.get('gripper', 0.0) for pt in trajectory])

    t_orig = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)

    pos_interp = CubicSpline(t_orig, positions, axis=0)(t_new)
    grip_interp = CubicSpline(t_orig, grippers)(t_new)
    r_interp = Slerp(t_orig, R.from_quat(quats))(t_new)

    smoothed = []

    for i in range(len(t_new) - 1):
        rot1 = r_interp[i]
        rot2 = r_interp[i + 1]

        # Compute angular difference in degrees
        relative_rot = rot1.inv() * rot2
        angle_deg = np.rad2deg(relative_rot.magnitude())

        # Subdivide if jump is large
        n_steps = max(1, int(np.ceil(angle_deg / angle_threshold_deg)))

        t_sub = np.linspace(0, 1, n_steps + 1)
        slerp_local = Slerp([0, 1], R.from_quat([rot1.as_quat(), rot2.as_quat()]))
        r_sub = slerp_local(t_sub)

        for j in range(n_steps):
            alpha = t_sub[j]
            p = (1 - alpha) * pos_interp[i] + alpha * pos_interp[i + 1]
            g = (1 - alpha) * grip_interp[i] + alpha * grip_interp[i + 1]
            angles = r_sub[j].as_euler('xyz', degrees=True)

            smoothed.append({
                'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
                'theta_x': float(angles[0]),
                'theta_y': float(angles[1]),
                'theta_z': float(angles[2]),
                'gripper': float(g)
            })

    # Append the final point
    final_angles = r_interp[-1].as_euler('xyz', degrees=True)
    smoothed.append({
        'x': float(pos_interp[-1][0]),
        'y': float(pos_interp[-1][1]),
        'z': float(pos_interp[-1][2]),
        'theta_x': float(final_angles[0]),
        'theta_y': float(final_angles[1]),
        'theta_z': float(final_angles[2]),
        'gripper': float(grip_interp[-1])
    })

    return smoothed

def replay_trajectory_smooth(base, trajectory):
    print("Replaying smoothed trajectory with action completion check...")
    for i, pt in enumerate(tqdm(trajectory, desc="Steps", unit="step")):
        success = execute_pose_and_wait(base, pt, index=i)
        if not success:
            print(f"[Warning] Action at step {i} did not complete.")
            return False

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = pt.get('gripper', 0.0) + 0.1
        base.SendGripperCommand(gripper_command)

        # --- Send pose command ---
        success = execute_pose_and_wait(base, pt, index=i)
        if not success:
            print(f"[Warning] Action at step {i} did not complete.")
            return False

    print("Trajectory execution complete.")
    return True

def run_sequence(base, base_cyclic, ip):
    success = True
    try:
        # 1. Move to home
        success &= example_move_to_home_position(base)

        # 2. Grasp
        grab_traj_path = os.path.join(os.path.dirname(__file__), "../generalized/failure/door_inversed_135/generalized_dmp_reaching_pose.json")
        success &= move_and_grab(base, grab_traj_path)

        # 3. Interpolate & replay
        traj_path = os.path.join(os.path.dirname(__file__), "../generalized/failure/door_inversed_135/generalized_dmp_quaternion_pose.json")
        with open(traj_path, 'r') as f:
            trajectory = json.load(f)
        smoothed_traj = interpolate_trajectory(trajectory, num_points=200)
        success &= replay_trajectory_smooth(base, smoothed_traj)

        return 0 if success else 1

    except (BrokenPipeError, TimeoutError, OSError) as e:
        print(f"[WARNING] Connection lost: {e}. Attempting reconnection...")

        try:
            _, _, base, base_cyclic = reconnect_robot(ip)
            return run_sequence(base, base_cyclic, ip)
        except Exception as e:
            print("[ERROR] Failed to reconnect:", e)
            return 1


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)
            return run_sequence(base, base_cyclic, args.ip)
    except (BrokenPipeError, TimeoutError, OSError) as e:
        print(f"[ERROR] Initial connection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
