from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import csv

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

#Create a function to get poses_3d keypoints
def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d

#Create a function to get xy coordinate for write csv file
def get2dxy(img, poses_2d):
    global b,v,pose,c
    for pose_id in range(len(poses_2d)):
        pose = np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)).transpose()
        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                v = (pose[0:1])
                b = (pose[1:2])
                c = np.dstack((v,b))
                c = np.resize(c,(1,38))
                #print(b)
                #print(v)
                #print(c)
                return c,v,pose
                
        

if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    
    stride = 8
    base_height = 256
    fx = -1
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    is_video = False
    
    from modules.inference_engine_pytorch import InferenceEnginePyTorch
    net = InferenceEnginePyTorch('human-pose-estimation-3d.pth', 'GPU')
    
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = None
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)
    
    for i in range(1, 10, 1):
        frame_provider = ImageReader([f'Fall ({i:d}).jpg'])
        print(i)
        for frame in frame_provider:
            current_time = cv2.getTickCount()
            if frame is None:
                break
            input_scale = base_height / frame.shape[0]
            scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
            scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]
            if fx < 0:  # Focal length is unknown
                fx = np.float32(0.8 * frame.shape[1])
                    
            inference_result = net.infer(scaled_img)
            poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
            edges = []
            if len(poses_3d):
                poses_3d = rotate_poses(poses_3d, R, t)
                poses_3d_copy = poses_3d.copy()
                x = poses_3d_copy[:, 0::4]
                y = poses_3d_copy[:, 1::4]
                z = poses_3d_copy[:, 2::4]
                poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
                poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
                    
            plotter.plot(canvas_3d, poses_3d, edges)
            cv2.imshow(canvas_3d_window_name, canvas_3d)
            draw_poses(frame, poses_2d)
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                       (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imshow('ICV 3D Human Pose Estimation', frame)
            #Save the processed image for bug check
            #cv2.imwrite(f'Stand0{i:d}.jpg', frame)
            get2dxy(frame,poses_2d)
            with open('123.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(c)