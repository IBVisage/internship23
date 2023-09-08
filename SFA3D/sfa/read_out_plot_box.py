import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import os

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration



def read_data_from_txt(path_to_file):
    array_2d_list = []
    array_3d_list = []

    current_array = None

    with open(path_to_file, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:
                current_array = None
                continue

            values = [float(value) for value in line.split('\t')]

            if current_array is None:
                if len(values) == 8:
                    current_array = array_2d_list
                else:
                    current_array = []  # Initialize a temporary list for 3D data

            current_array.append(values)

            if len(current_array) == 8:
                array_3d_list.append(current_array)  # Append the 3D data as a whole

    array_2d = np.array(array_2d_list)
    array_3d = np.array(array_3d_list)

    return array_2d, array_3d


video_num = input("Input the video number (00 - 28) : ")

# Set the current directory as the working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory + "/outputs/outputs_video_" + video_num)

# Define the file name (assuming it's in the current directory)
file_name = 'out_000000.txt'  # Replace with the actual file name
outputs, corners = read_data_from_txt(file_name)
print("2D Array:")
print(outputs)
print("3D Array:")
print(corners)

print("\n\n")
print(outputs.shape)
print(corners.shape)
print("\n\n")

calib = Calibration("../dataset/kitti/testing/calib/00" + video_num + ".txt")

if len(outputs) > 0:
    outputs[:, 1:] = lidar_to_camera_box(outputs[:, 1:], calib.V2C, calib.R0, calib.P2)
    img_bgr, corners_out = show_rgb_image_with_boxes(img_bgr, outputs, calib)

print(corners_out)