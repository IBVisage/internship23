import os
import numpy as np
import tracking_utils.kalman_utils as ukf
import matplotlib.pyplot as plt
import math


# # Initialize the plot
# fig, ax = plt.subplots()

# Set the print options for NumPy
np.set_printoptions(precision=3, suppress=True)

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


def plot_objects(object_data, ids_data, orientation_data, ax):
    ax.clear()

    for obj_id, obj in enumerate(object_data):
        x = obj[0]
        y = obj[3]
        width = orientation_data[obj_id][0]
        length = orientation_data[obj_id][1]
        orientation = orientation_data[obj_id][2]
        identification = ids_data[obj_id]

        # Calculate the coordinates of the four corners of the box
        x1 = x - length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y1 = y - length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x2 = x + length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y2 = y + length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x3 = x + length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y3 = y + length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)
        x4 = x - length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y4 = y - length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)

        # Define the vertices of each edge
        front_edge = [[x1, y1], [x2, y2]]
        left_edge = [[x1, y1], [x4, y4]]
        right_edge = [[x2, y2], [x3, y3]]
        back_edge = [[x3, y3], [x4, y4]]

        # Plot each edge with the specified colors
        ax.plot(*zip(*front_edge), color='red', linewidth=2)
        ax.plot(*zip(*left_edge), color='red', linewidth=2)
        ax.plot(*zip(*right_edge), color='lightblue', linewidth=2)
        ax.plot(*zip(*back_edge), color='red', linewidth=2)

        # Add the object ID as text near the object center
        obj_center_x = (x1 + x3) / 2
        obj_center_y = (y1 + y3) / 2
        ax.text(obj_center_x, obj_center_y, str(identification), color='black', ha='center', va='center', fontsize=12)

    # Set axis limits and labels
    ax.set_xlim(0, 50)  # Adjust the x-axis limits as needed
    ax.set_ylim(25, -25)  # Adjust the y-axis limits as needed
    ax.set_xlabel('X-axis')  # Label the x-axis as X
    ax.set_ylabel('Y-axis')  # Label the y-axis as Y

    # Draw the plot
    plt.gca().invert_yaxis()  # Invert the y-axis if needed
    plt.draw()
    plt.pause(0.25)  # Pause for 1 second (you can adjust this time as needed)


# def on_key(event):
#     global objects_list, ids_list, orientations_list
#
#     for index, frame_object in objects_list:
#         if event.key == 'right':
#             plot_objects(frame_object, ids_list[index], orientations_list[index])


# Folder containing detection files
detections_folder = "./outputs/out_vid_polar_all/out_vid_12"

# List all files in the detections folder
detection_files = os.listdir(detections_folder)

# Sort the files to ensure they are processed in order
detection_files.sort()
print(detection_files)

max_frames_lost = 7  # Maximum frames to consider a track lost
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
current_object_data = []
current_object_orientations = []
current_object_ids = []

for track_id, active_track in active_tracks.items():
    current_object_data.append(active_track.current_prediction)
    current_object_ids.append(track_id)
    current_object_orientations.append(active_track.current_object[5:8])

# ili object_data_list.append(current_object_data)
objects_list.append(current_object_data)
ids_list.append(current_object_ids)
orientations_list.append(current_object_orientations)


# Iterate through detection files and load detections
for frame, detection_file in enumerate(detection_files, start=1):
    detection_file_path = os.path.join(detections_folder, detection_file)

    current_object_data = []
    current_object_orientations = []
    current_object_ids = []

    # Load detections from the file
    detections = np.loadtxt(detection_file_path, delimiter='\t')  # Adjust delimiter as needed

    if frame > 1:
        tracks_to_update = []

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

        # print(f"\nDetections in frame {frame - 1} are {detections}\n")
        # Increment the age of all active tracks
        for act_track in active_tracks.values():
            act_track.increment_age()

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
                        tracks_to_update.append(min_track_id)
                    else:
                        new_track_id = len(all_tracks) + 1
                        make_new_track(new_track_id, detection)
                        matched_detections.add(new_track_id)

        print(f"\nFrame je {frame}\n")
        for active_track_id, active_track in active_tracks.items():
            print(f"ID trake : {active_track_id}, Frames since last update : {active_track.frames_since_last_update}")

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

    for track_id, active_track in active_tracks.items():
        current_object_data.append(active_track.current_prediction)
        current_object_ids.append(track_id)
        current_object_orientations.append(active_track.current_object[5:8])

    # ili object_data_list.append(current_object_data)
    objects_list.append(current_object_data)
    ids_list.append(current_object_ids)
    orientations_list.append(current_object_orientations)

# # Print the final list of all tracks
# print("\nAll Tracks:")
# for track in all_tracks:
#     print(
#         f"Track {track.track_id}: Trajectory Length: {len(track.previous_objects) + 1}\n "
#         f"Previous objects: {track.previous_objects} \n Current object: {track.current_object}")

# # Connect the key press event
# fig.canvas.mpl_connect('key_press_event', on_key)
#
# # Plot the initial objects
# plot_objects(current_object_data, current_object_ids, current_object_orientations)
#
# # Show the plot
# plt.show()

print(f"Objekti su: {objects_list[0]}")
print(f"Ids su : {ids_list[0]}")
print(f"Orientations su : {orientations_list[0]}")


# Create a single figure and axis outside the loop
fig, ax = plt.subplots()

for i in range(len(objects_list)):
    plot_objects(objects_list[i], ids_list[i], orientations_list[i], ax)
    plt.pause(0.25)  # Pause for 1 second between plots

plt.show()
