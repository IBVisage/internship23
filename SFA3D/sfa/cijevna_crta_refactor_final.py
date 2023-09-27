"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

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
from copy import copy

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values, draw_real_to_bev, draw_connecting_line
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes, draw_box_rgb_prediction
from data_process.kitti_data_utils import Calibration
from hungarian import testing_function



import os
import numpy as np
import tracking_utils.kalman_utils as ukf
import tracking_utils.visual_utils as visuals
import tracking_utils.metric_utils as metric
import matplotlib.pyplot as plt
from tracking_utils.coordinate_transform import cart_2_polar


np.set_printoptions(precision=3, suppress=True)


dt = 0.1
sigma_a = 0.1
F = np.array([[1, dt, 0.5*(dt**2), 0, 0, 0],
              [0, 1, dt, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*(dt**2)],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])

Q = (sigma_a**2) * np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                            [dt**3/2, dt**2, dt, 0, 0, 0],
                            [dt**2/2, dt, 1, 0, 0, 0],
                            [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                            [0, 0, 0, dt**3/2, dt**2, dt],
                            [0, 0, 0, dt**2/2, dt, 1]])

sigma_r = 4
sigma_phi = 0.1
R = np.array([[sigma_r**2, 0],
              [0, sigma_phi**2]])

W = np.array([[-1], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6]])
W_for_diag = [-1, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
W_mat = np.diag(W_for_diag)

P = np.array([[10, 0, 0, 0, 0, 0],
              [0, 10, 0, 0, 0, 0],
              [0, 0, 10, 0, 0, 0],
              [0, 0, 0, 10, 0, 0],
              [0, 0, 0, 0, 10, 0],
              [0, 0, 0, 0, 0, 10]])

class Track:
    def __init__(self, identification, initial_state, cov_matrix):
        self.track_id = identification
        self.current_object = initial_state
        self.previous_objects = []
        self.frames_since_last_update = 0
        self.current_prediction = np.array([self.current_object[1], 0, 0, self.current_object[2], 0, 0])
        self.current_estimate = np.array([self.current_object[1], 0, 0, self.current_object[2], 0, 0])
        self.covariance_matrix_prediction = cov_matrix
        self.covariance_matrix_estimate = self.covariance_matrix_prediction
        self.propagated_sigma_matrix = None

    def update(self, new_object):
        self.previous_objects.append(self.current_object)
        self.current_object = new_object
        self.frames_since_last_update = 0

    def increment_age(self):
        self.frames_since_last_update += 1

    def is_lost(self, max_frames_inactive):
        return self.frames_since_last_update >= max_frames_inactive

    def get_trajectory(self):
        return np.vstack([self.previous_objects, self.current_object])
    
def make_new_track_ref(identification, detection_vec):
        new_track = Track(identification, detection_vec, P)
        active_tracks_ref[new_track_id] = new_track
        return None

    
def predict_all_active_tracks_ref():
    if len(active_tracks_ref) > 0:
        for active_tracks_id, active_track in active_tracks_ref.items():
            if active_track.frames_since_last_update > 0:
                x_pred, p_pred, sigma = ukf.predict(F, active_track.covariance_matrix_estimate, Q,
                                                    active_track.current_prediction, W, W_mat)
            else:
                x_pred, p_pred, sigma = ukf.predict(F, active_track.covariance_matrix_estimate, Q,
                                                    active_track.current_estimate, W, W_mat)

            active_track.current_prediction = x_pred
            active_track.covariance_matrix_prediction = p_pred
            active_track.propagated_sigma_matrix = sigma

    return None

def update_all_active_tracks_ref():
    if len(active_tracks_ref) > 0:
        for active_tracks_id, active_track in active_tracks_ref.items():
            if active_track.frames_since_last_update == 0 and active_tracks_id in tracks_to_update:
                x_estimate, p_estimate = ukf.update(F, active_track.covariance_matrix_prediction, Q, R,
                                                    active_track.current_prediction, W, W_mat,
                                                    active_track.propagated_sigma_matrix,
                                                    active_track.current_object[-2:], active_tracks_id)

                active_track.current_estimate = x_estimate
                active_track.covariance_matrix_estimate = p_estimate

    return None


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    cons = 0
    video_num = input("Broj videa iz KITTI tracking dataseta (00-28) : ")
    if len(video_num) == 1:
        video_num = "000"+video_num
    else:
        video_num = "00"+video_num
    out_cap = None
    model.eval()
    iteration = 0


    ### REFACTORING
    max_frames_lost = 5


    active_tracks_ref = {}
    num_of_all_tracks = 0

    ### REFACTORING

    test_dataloader = create_test_dataloader(configs, video_num)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]  # only first batch
            # Draw prediction in the image
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)

            img_path = metadatas['img_path'][0]
            img_rgb = img_rgbs[0].numpy()
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            calib = Calibration("../dataset/kitti/testing/calib/" + video_num + ".txt")
            # Za zaustavljanje slika da se može video kreirati
            max_frames = len(os.listdir("../dataset/kitti/testing/image_2_all/" + video_num))

            kitti_dets = convert_det_to_real_values(detections)
            kitti_dets_copius = np.copy(kitti_dets)

            ### REFACTOR POČETAK KALMANA
            # Dohvaćanje i pretvaranje detekcija u polarne
            kitti_dets_copius = cart_2_polar(kitti_dets_copius)
            # Stvaranje prvih traka
            if iteration == 0:
                for dt in kitti_dets_copius:
                    num_of_all_tracks += 1
                    new_track_id = num_of_all_tracks
                    make_new_track_ref(new_track_id, dt)
                pass
                # Prvotno predviđanje
                predict_all_active_tracks_ref()

            # Početak rada kalmana
            if iteration >= 1:
                hung_threshold = 7.5
                hung_ignore = 1
                # Hungarian auction dodijeljivanje parova
                used_track_detection_pairs, unused_tracks, unused_detections = testing_function(copy(kitti_dets_copius), copy(active_tracks_ref),
                                                                                        metric.k_iou_euc, hung_threshold,
                                                                                        hung_ignore, iteration)
                


                tracks_to_update = []
                tracks_to_remove = []

                for act_track in active_tracks_ref.values():
                    act_track.increment_age()

                # Guard da se objekti pravilno stvore
                if kitti_dets_copius.ndim == 1:
                    if len(kitti_dets_copius) == 0:
                        kitti_dets_copius = np.empty((0, 10))
                    else:
                        kitti_dets_copius = np.array([kitti_dets_copius])
                # Generiranje detekcija koje nisu dobile dodijeljenu traku
                for det in unused_detections:
                    if len(det):
                        num_of_all_tracks += 1
                        new_track_id = num_of_all_tracks
                        make_new_track_ref(new_track_id, det)
                    
                # Postavljanje traka za update onih kojima je nađen par
                for track_det_pair in used_track_detection_pairs:
                    active_tracks_ref[track_det_pair[0]].update(track_det_pair[1])
                    active_tracks_ref[track_det_pair[0]].frames_since_last_update = 0
                    tracks_to_update.append(track_det_pair[0])
                    pass

                update_all_active_tracks_ref()

                # Sada imamo sve detekcije, sve trake. Povezane trake i detekcije su u used_track_detection_pairs i njih
                # treba povezati na slici.
                for used_pair in used_track_detection_pairs:
                    # Stvaranje kopija da ne bude problema u funkcijama. Neke funkcije isto stvaraju kopije na početku
                    track = copy(active_tracks_ref[used_pair[0]])
                    pair_track_id = copy(used_pair[0])
                    pair_detection = copy(used_pair[1])
                    track_prediction = track.current_prediction
                    track_object = track.current_object

                    # Konstrukcija objekta
                    pair_track_object = np.zeros((8, ))
                    pair_track_object[0] = pair_detection[0]
                    pair_track_object[1] = track_prediction[0]
                    pair_track_object[2] = track_prediction[3]
                    pair_track_object[3:5] = track_object[3:5]
                    pair_track_object[5:8] = track_object[5:8]
                    # Crtanja na sliku i bev                    
                    draw_real_to_bev(pair_track_object, bev_map, pair_track_id)
                    draw_connecting_line(bev_map, pair_track_object, pair_detection)

                    pair_track_object[1:] = lidar_to_camera_box([pair_track_object[1:]], calib.V2C, calib.R0, calib.P2)
                    img_bgr = draw_box_rgb_prediction(img_bgr, pair_track_object, calib, pair_track_id, 0)
                    
                
                # Crtanje traka koje nemaju povezanu detekciju
                for unused_track_id in unused_tracks:
                    if unused_track_id not in tracks_to_remove:
                        unused_track = active_tracks_ref[unused_track_id]
                        
                        draw_track = copy(unused_track.current_object[0:8])

                        draw_real_to_bev(draw_track, bev_map, unused_track.track_id)

                        draw_track[1:] = lidar_to_camera_box([draw_track[1:]], calib.V2C, calib.R0, calib.P2)
                        img_bgr = draw_box_rgb_prediction(img_bgr, draw_track, calib, unused_track.track_id, 0)


                predict_all_active_tracks_ref()

                # Uklanjanje nepotrebnih traka
                for track_id, track in active_tracks_ref.items():
                    if track.is_lost(max_frames_lost):
                        tracks_to_remove.append(track_id)

                for track_id in tracks_to_remove:
                    del active_tracks_ref[track_id]

            ### REFACTOR KRAJ

            iteration += 1

            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr, corners_out = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
            
            ### Pisanje u datoteke
            # # Create a folder for outputs_video if it doesn't exist
            # output_folder = "outputs_video_" + video_num
            # if not os.path.exists("outputs/" + output_folder):
            #     os.makedirs("outputs/" + output_folder)

            # formatted_cons = f"{cons:06d}"
            # # Define file name for the combined data
            # combined_filename = os.path.join("outputs/" + output_folder, "out_" + formatted_cons + ".txt")

            """ Ispod je dio za sačuvavanje podataka iz videa. """

            # print(kitti_dets_copius)
            # print(corners_out)

            # # Save kitti_dets_copius to a text file
            # with open(combined_filename, "w") as file:
            #     np.savetxt(file, kitti_dets_copius, fmt='%.8f', delimiter='\t')
            #     file.write("\n")  # Add a blank line
            #
            #     # Loop through corners_2d_list and save each set as a separate 2D array
            #     for i, corners_2d in enumerate(corners_out):
            #         set_filename = f"corners_2d_set_{i}.txt"
            #         np.savetxt(file, corners_2d, fmt='%d', delimiter='\t')
            #         file.write("\n")  # Add a blank line between sets
            #
            # print(f'Data saved successfully in {combined_filename}.')

            cons = cons + 1

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(video_num)),
                            #os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            # cv2.imshow('test-img', out_img)
            # print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            # if cv2.waitKey(0) & 0xFF == 27:
            #     break
            
            # Test da ne pukne spisivanje slika
            if iteration==max_frames:
                break
    if out_cap:
        out_cap.release()


    cv2.destroyAllWindows()
