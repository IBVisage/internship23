import os
import numpy as np
import kalman_utils as ukf
import matplotlib.pyplot as plt
import math


# Initialize the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Initialize Kalman filter data
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

# Initialize containers for tracks
active_tracks = {}  # Dictionary for active tracks
all_tracks = []  # List to keep track of all tracks


# Here, current_object is the data of object after the estimate (in frame > 1)!
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


# Here, current_prediction is the last prediction, and current_object is the measurement after the association!
def update_all_active_tracks():
    if len(active_tracks) > 0:
        for active_tracks_id, active_track in active_tracks.items():
            if active_track.frames_since_last_update == 1:
                x_estimate, p_estimate = ukf.update(F, active_track.covariance_matrix_prediction, Q, R,
                                                    active_track.current_prediction, W, W_mat,
                                                    active_track.propagated_sigma_matrix,
                                                    active_track.current_object[-2:])

                active_track.current_estimate = x_estimate
                active_track.covariance_matrix_estimate = p_estimate

    return None


def make_new_track(identification, detection_vec):
    new_track = Track(identification, detection_vec, P)
    active_tracks[new_track_id] = new_track
    all_tracks.append(new_track)
    return None


# Custom cosine distance function
def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 1.0  # Handle division by zero

    return 1.0 - (dot_product / (norm_vec1 * norm_vec2))


class Track:
    def __init__(self, identification, initial_state, cov_matrix):
        self.track_id = identification
        self.current_object = initial_state
        self.previous_objects = []  # List to store previous object states
        self.frames_since_last_update = 0
        self.current_prediction = np.array([self.current_object[1], 0, 0, self.current_object[2], 0, 0])
        self.current_estimate = np.array([self.current_object[1], 0, 0, self.current_object[2], 0, 0])
        self.covariance_matrix_prediction = cov_matrix
        self.covariance_matrix_estimate = self.covariance_matrix_prediction
        self.propagated_sigma_matrix = None

    def update(self, new_object):
        self.previous_objects.append(self.current_object)
        self.current_object = new_object
        self.frames_since_last_update = 0  # Reset age when updated

    def increment_age(self):
        self.frames_since_last_update += 1

    def is_lost(self, max_frames_inactive):
        return self.frames_since_last_update >= max_frames_inactive

    def get_trajectory(self):
        # Get the full trajectory of the object, including current and previous states
        return np.vstack([self.previous_objects, self.current_object])


# Folder containing detection files
detections_folder = "./out_vid_polar_all/out_vid_12"

# List all files in the detections folder
detection_files = os.listdir(detections_folder)

# Sort the files to ensure they are processed in order
detection_files.sort()
print(detection_files)

max_frames_lost = 5  # Maximum frames to consider a track lost
distance_threshold = 10  # Threshold for associating a detection with a track

# Initial frame stuff
initial_frame = np.loadtxt(os.path.join(detections_folder, "out_000000.txt"), delimiter='\t')

for detection in initial_frame:
    new_track_id = len(all_tracks) + 1
    make_new_track(new_track_id, detection)

predict_all_active_tracks()

objects_list = []
ids_list = []
orientations_list = []

# Initialize object index and current object data
object_index = 0
current_object_data = []
current_object_orientations = []
current_object_ids = []

for track_id, active_track in active_tracks:
    current_object_data.append(active_track.current_prediction)
    current_object_ids.append(track_id)
    current_object_orientations.append(active_track.current_object[7])

# ili object_data_list.append(current_object_data)
objects_list[object_index] = current_object_data
ids_list[object_index] = current_object_ids
orientations_list[object_index] = current_object_orientations
object_index += 1

# Iterate through detection files and load detections
for frame, detection_file in enumerate(detection_files, start=1):
    detection_file_path = os.path.join(detections_folder, detection_file)

    # Load detections from the file
    detections = np.loadtxt(detection_file_path, delimiter='\t')  # Adjust delimiter as needed

    if frame > 1:
        # Ensure detections is a 2D array with the second dimension as 10
        if detections.ndim == 1:
            if len(detections) == 0:
                detections = np.empty((0, 10))  # Empty 2D array
            else:
                detections = np.array([detections])

        # List to keep track of detections that have been associated with a track
        matched_detections = set()

        # List to keep track IDs that should be removed
        tracks_to_remove = []

        # Increment the age of all active tracks
        for act_track in active_tracks.values():
            act_track.increment_age()

        # print(f"\nDetections in frame {frame - 1} are {detections}\n")

        # Make sure that loop is not entered if there are no detections this frame
        if len(detections) > 0:
            # Process each detection
            for detection in detections:
                # distances = np.array(len(active_tracks.values()))
                min_distance = 50
                min_track_id = -1

                for act_track in active_tracks.values():
                    distance_comparison_element = act_track.current_object
                    distance_comparison_element[1] = act_track.current_prediction[0]
                    distance_comparison_element[2] = act_track.current_prediction[3]
                    # distance = cosine_distance(detection, distance_comparison_element)
                    distance = np.linalg.norm(detection[1:3] - distance_comparison_element[1:3])

                    # print(f"Distance koji nas sekira je {distance}")
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
                    else:
                        new_track_id = len(all_tracks) + 1
                        make_new_track(new_track_id, detection)
                        matched_detections.add(new_track_id)

        print(f"\nFrame je {frame}\n")
        for active_track in active_tracks:
            print(active_track.frames_since_last_update)
        update_all_active_tracks()

        # Identify tracks to be removed from the active_tracks dictionary
        for track_id, track in active_tracks.items():
            if track.is_lost(max_frames_lost):
                tracks_to_remove.append(track_id)

        # Remove lost tracks from the active_tracks dictionary
        for track_id in tracks_to_remove:
            del active_tracks[track_id]

        predict_all_active_tracks()

        # Print active track IDs for this frame
        # print(f"Frame {frame}: Active Track IDs: {list(active_tracks.keys())}")

# # Print the final list of all tracks
# print("\nAll Tracks:")
# for track in all_tracks:
#     print(
#         f"Track {track.track_id}: Trajectory Length: {len(track.previous_objects) + 1}\n "
#         f"Previous objects: {track.previous_objects} \n Current object: {track.current_object}")

""" 
Please
stop
embarassing
yourself
mate
.
"""

def on_key(event):
    global object_index, object_data_list, current_object_data

    if event.key == 'right':  # Right arrow key
        # Implement your algorithm to generate new object data here
        # For this example, we will just rotate the existing objects
        new_object_data = []

        object_data_list.append(new_object_data)

        # Update the object index and current object data
        object_index += 1
        current_object_data = object_data_list[object_index]

        # Plot the new objects
        plot_objects(current_object_data)

""" Do ovdje """

""" Sada ide implementacija on_key funkcije """



""" Do ovdje """

""" Sada ispod ide ovaj final blow """
# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Plot the initial objects
plot_objects(current_object_data)

# Show the plot
plt.show()