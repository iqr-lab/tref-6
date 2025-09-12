#! /usr/bin/env python3

import sys
import os
import time
import json
import pygame
import cv2
import numpy as np
import pyrealsense2 as rs

from datetime import datetime

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import BaseCyclic_pb2
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceManager_pb2



#TIMEOUT_DURATION = 20
GRIPPER_INCREMENT = 0.1
POSE_LOG =[]
FRAME_LOG = []

pipeline = None
video_writer = None
first_frame = None
first_depth = None
align = None


# Replace these with your actual intrinsics
fx, fy = 615.0, 615.0
cx, cy = 320.0, 240.0





def init_camera():
    global pipeline, fx, fy, cx, cy, align
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    # Get intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Save intrinsics to file
    intrinsics_file = "camera_intrinsics.txt"
    with open(intrinsics_file, "w") as f:
        f.write(f"width: {intrinsics.width}\n")
        f.write(f"height: {intrinsics.height}\n")
        f.write(f"fx: {fx}\n")
        f.write(f"fy: {fy}\n")
        f.write(f"cx: {cx}\n")
        f.write(f"cy: {cy}\n")
        f.write(f"distortion_model: {intrinsics.model}\n")
        f.write(f"coeffs: {intrinsics.coeffs}\n")

    print(f"[INFO] Camera intrinsics saved to {intrinsics_file}")


def stop_camera():
    global pipeline, video_writer
    if video_writer:
        video_writer.release()
    if pipeline:
        pipeline.stop()
    print("[INFO] Camera stopped.")

def capture_rgb_frame():
    global first_frame, first_depth, align
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    if first_frame is None:
        first_frame = color_image.copy()
        first_depth = depth_image.copy()
    return color_image

def enable_gravity_compensation(base):

    base_admittance_mode = Base_pb2.Admittance()
    base_admittance_mode.admittance_mode = Base_pb2.CARTESIAN

    try:
        base.SetAdmittance(base_admittance_mode)
    except Exception as e:
        print("Failed")
        return False
    print("Gravity Compensation Mode")
    return True

def disable_gravity_compensation(base):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    print("Exit Gravity Compensation Mode")

def list_connected_devices(device_manager):
    devices = device_manager.ReadAllDevices()
    print("[INFO] Connected devices:")
    for d in devices.device_handle:
        dev_type = DeviceManager_pb2.DeviceTypes.Name(d.device_type)
        print(f" - {dev_type} (ID: {d.device_identifier})")



def send_gripper_command(base, value):
    #disable_gravity_compensation(base)
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_SPEED
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = value
    base.SendGripperCommand(gripper_command)
    #enable_gravity_compensation(base)


def log_pose(base, base_cyclic):
    try:
        pose = base.GetMeasuredCartesianPose()
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)

        gripper_value = gripper_measure.finger[0].value if len(gripper_measure.finger) > 0 else None

        POSE_LOG.append({
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "theta_x": pose.theta_x,
            "theta_y": pose.theta_y,
            "theta_z": pose.theta_z,
            "gripper": gripper_value
        })
    except Exception as e:
        print("Failed to get pose: ", e)

def save_trajectory_to_file():
    global first_frame, first_depth, video_writer

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save trajectory
    traj_file = f"trajectory_{timestamp}.json"
    with open(traj_file, "w") as f:
        json.dump(POSE_LOG, f, indent=4)
    print(f"Trajectory saved to {traj_file}")

    # Save first frame
    first_frame_file = f"first_frame_{timestamp}.jpg"
    if first_frame is not None:
        cv2.imwrite(first_frame_file, first_frame)
        print(f"First frame saved to {first_frame_file}")
    
    first_depth_npy = f"first_depth_{timestamp}.npy"
    first_depth_png = f"first_depth_{timestamp}.png"
    if first_depth is not None:
        np.save(first_depth_npy, first_depth)
        print(f"First Depth frame saved to {first_depth_npy}")
        cv2.imwrite(first_depth_png, first_depth)
        print(f"First Depth visualization saved to {first_depth_png}")



def pygame_control_loop(base, base_cyclic, device_manager):
    # global video_writer

    pygame.init()
    screen = pygame.display.set_mode((500,300))
    pygame.display.set_caption("Kinova Gen3 Gravity Compensation Control")

    font = pygame.font.Font(None, 36)
    running = True
    disable_gravity_compensation(base)
    gravity_mode = False

    init_camera()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #video_writer = cv2.VideoWriter(f"rgb_{timestamp}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))


    while running:
        screen.fill((30, 30, 30))
        text = font.render(
            "Press G to Enable | E to Disable | C to close gripper | X to open gripper | Q to Quit", True, (255, 255, 255)
        )
        screen.blit(text, (20, 130))

        pygame.display.flip()
        keys = pygame.key.get_pressed()
        gripper = (keys[pygame.K_c] - keys[pygame.K_x]) * GRIPPER_INCREMENT


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    if not gravity_mode:
                        success = enable_gravity_compensation(base)
                        if success:
                            gravity_mode = True
                elif event.key == pygame.K_e:
                    if gravity_mode:
                        disable_gravity_compensation(base)
                        gravity_mode = False
                elif event.key == pygame.K_q:
                    running = False

        rgb_frame = capture_rgb_frame()
        # if rgb_frame is not None:
        #     video_writer.write(rgb_frame)

        if gravity_mode:
            log_pose(base, base_cyclic)
        else:
            if any([gripper]):
                send_gripper_command(base, gripper)
            else:
                send_gripper_command(base, 0.0)

        time.sleep(0.1)

    save_trajectory_to_file()
    stop_camera()
    pygame.quit()

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    args = utilities.parseConnectionArguments()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        device_manager = DeviceManagerClient(router)
        list_connected_devices(device_manager)
        pygame_control_loop(base, base_cyclic, device_manager)

    


if __name__ == "__main__":
    exit(main())