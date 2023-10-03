import os
import numpy as np
import tracking_utils.kalman_utils as ukf
import tracking_utils.visual_utils as visuals
import tracking_utils.metric_utils as metric
import matplotlib.pyplot as plt

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

sigma_r = 0.1
sigma_phi = 0.01
R = np.array([[sigma_r**2, 0],
              [0, sigma_phi**2]])

W = np.array([[-1], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6]])
W_for_diag = [-1, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
W_mat = np.diag(W_for_diag)

P = np.array([[50, 0, 0, 0, 0, 0],
              [0, 50, 0, 0, 0, 0],
              [0, 0, 50, 0, 0, 0],
              [0, 0, 0, 50, 0, 0],
              [0, 0, 0, 0, 50, 0],
              [0, 0, 0, 0, 0, 50]])

active_tracks = {}
all_tracks = []


def predict_all_active_tracks():
    if len(active_tracks) > 0:
        for active_tracks_id, active_track in active_tracks.items():
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


def update_all_active_tracks():
    if len(active_tracks) > 0:
        for active_tracks_id, active_track in active_tracks.items():
            if active_track.frames_since_last_update == 0 and active_tracks_id in tracks_to_update:
                x_estimate, p_estimate = ukf.update(F, active_track.covariance_matrix_prediction, Q, R,
                                                    active_track.current_prediction, W, W_mat,
                                                    active_track.propagated_sigma_matrix,
                                                    active_track.current_object[-2:], active_tracks_id)

                active_track.current_estimate = x_estimate
                active_track.covariance_matrix_estimate = p_estimate

    return None


def make_new_track(identification, detection_vec):
    new_track = Track(identification, detection_vec, P)
    active_tracks[new_track_id] = new_track
    all_tracks.append(new_track)
    return None


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


detections_folder = "./outputs/out_vid_polar_all/out_vid_12"

detection_files = os.listdir(detections_folder)

detection_files.sort()

max_frames_lost = 5
distance_threshold = 10

initial_frame = np.loadtxt(os.path.join(detections_folder, "out_000000.txt"), delimiter='\t')

for detection in initial_frame:
    new_track_id = len(all_tracks) + 1
    make_new_track(new_track_id, detection)

predict_all_active_tracks()

objects_list = []
ids_list = []
orientations_list = []
vehicle_list = []

current_object_data = []
current_object_orientations = []
current_object_ids = []
current_object_vehicle = []

for track_id, active_track in active_tracks.items():
    current_object_data.append(active_track.current_prediction)
    current_object_ids.append(track_id)
    current_object_orientations.append(active_track.current_object[5:8])
    current_object_vehicle.append(active_track.current_object[0:5])

objects_list.append(current_object_data)
ids_list.append(current_object_ids)
orientations_list.append(current_object_orientations)
vehicle_list.append(current_object_vehicle)

for frame, detection_file in enumerate(detection_files, start=1):
    detection_file_path = os.path.join(detections_folder, detection_file)

    current_object_data = []
    current_object_orientations = []
    current_object_ids = []
    current_object_vehicle = []

    detections = np.loadtxt(detection_file_path, delimiter='\t')

    if frame > 1:
        tracks_to_update = []

        if detections.ndim == 1:
            if len(detections) == 0:
                detections = np.empty((0, 10))
            else:
                detections = np.array([detections])

        matched_detections = set()

        tracks_to_remove = []

        for act_track in active_tracks.values():
            act_track.increment_age()

        if len(detections) > 0:
            for detection in detections:
                min_distance = 50
                min_track_id = -1

                for act_track in active_tracks.values():
                    distance_comparison_element = act_track.current_object
                    distance_comparison_element[1] = act_track.current_prediction[0]
                    distance_comparison_element[2] = act_track.current_prediction[3]

                    distance = metric.euclidian(detection, distance_comparison_element)

                    if distance < min_distance:
                        min_distance = distance
                        min_track_id = act_track.track_id

                if min_distance >= distance_threshold or len(active_tracks) < 1:
                    new_track_id = len(all_tracks) + 1
                    make_new_track(new_track_id, detection)
                    matched_detections.add(new_track_id)

                if min_distance < distance_threshold:
                    if min_track_id not in matched_detections:
                        active_tracks[min_track_id].update(detection)
                        active_tracks[min_track_id].frames_since_last_update = 0
                        matched_detections.add(min_track_id)
                        tracks_to_update.append(min_track_id)
                    else:
                        new_track_id = len(all_tracks) + 1
                        make_new_track(new_track_id, detection)
                        matched_detections.add(new_track_id)

        print(f"\nFrame je {frame}\n")
        for active_track_id, active_track in active_tracks.items():
            print(f"ID trake : {active_track_id}, Frames since last update : {active_track.frames_since_last_update}")

        update_all_active_tracks()

        for track_id, track in active_tracks.items():
            if track.is_lost(max_frames_lost):
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del active_tracks[track_id]

        predict_all_active_tracks()

    for track_id, active_track in active_tracks.items():
        current_object_data.append(active_track.current_prediction)
        current_object_ids.append(track_id)
        current_object_orientations.append(active_track.current_object[5:8])
        current_object_vehicle.append(active_track.current_object[0:5])

    objects_list.append(current_object_data)
    ids_list.append(current_object_ids)
    orientations_list.append(current_object_orientations)
    vehicle_list.append(current_object_vehicle)

# # Create a single figure and axis outside the loop
# fig, ax = plt.subplots()
#
# for i in range(len(objects_list)):
#     visuals.plot_objects(objects_list[i], ids_list[i], orientations_list[i], ax)
#     plt.pause(0.01)  # Pause for 1 second between plots
#
# plt.show()