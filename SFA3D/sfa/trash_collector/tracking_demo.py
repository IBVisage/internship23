import os
import numpy as np

# Initialize containers for tracks
active_tracks = {}  # Dictionary for active tracks
all_tracks = []  # List to keep track of all tracks


def ukf_predict(arg1, arg2):
    return 1


def ukf_update(arg1, arg2, arg3):
    return 1


# Here, current_object is the data of object after the estimate (in frame > 1)!
def predict_all_active_tracks():
    if len(active_tracks) > 1:
        for active_tracks_id, active_track in active_tracks.items():
            if active_track.frames_since_last_update > 1:
                active_track.current_prediction, cov_mat = ukf_predict(active_track.current_prediction,
                                                                       active_track.current_covariance_matrix)
            else:
                active_track.current_prediction, cov_mat = ukf_predict(active_track.current_object,
                                                                       active_track.current_covariance_matrix)
    return None


# Here, current_prediction is the last prediction, and current_object is the measurement after the association!
def update_all_active_tracks():
    if len(active_tracks) > 1:
        for active_tracks_id, active_track in active_tracks.items():
            if active_track.frames_since_last_update == 1:
                active_track.current_object, cov_mat = ukf_update(active_track.current_object,
                                                                  active_track.current_prediction,
                                                                  active_track.current_covariance_matrix)
    return None


def make_new_track(identification):
    new_track = Track(identification, detection)
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
        self.s = []  # List to store previous object states
        self.frames_since_last_update = 0
        self.current_prediction = self.current_object
        self.covariance_matrices = []
        self.current_covariance_matrix = cov_matrix

    def update(self, new_object):
        self.previous_objects.append(self.current_object)
        self.current_object = new_object
        self.frames_since_last_update = 0  # Reset age when updated

    def update_covariance_matrix(self, new_matrix):
        self.covariance_matrices.append(self.current_covariance_matrix)
        self.current_covariance_matrix = new_matrix

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
distance_threshold = 0.1  # Threshold for associating a detection with a track

# Initial frame stuff
initial_frame = np.loadtxt(os.path.join(detections_folder, "out_000000.txt"), delimiter='\t')

for detection in initial_frame:
    new_track_id = len(all_tracks) + 1
    make_new_track(new_track_id)

predict_all_active_tracks()

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

        print(f"\nDetections in frame {frame} are {detections}\n")

        # Make sure that loop is not entered if there are no detections this frame
        if len(detections) > 0:
            # Process each detection
            for detection in detections:
                # distances = np.array(len(active_tracks.values()))
                min_distance = 1
                min_track_id = -1

                for act_track in active_tracks.values():
                    distance = cosine_distance(detection, act_track.current_prediction)
                    if distance < min_distance:
                        min_distance = distance
                        min_track_id = act_track.track_id

                if min_distance >= distance_threshold or len(active_tracks) < 1:
                    new_track_id = len(all_tracks) + 1
                    make_new_track(new_track_id)
                    matched_detections.add(new_track_id)

                if min_distance < distance_threshold:
                    if min_track_id not in matched_detections:
                        active_tracks[min_track_id].update(detection)
                        active_tracks[min_track_id].frames_since_last_update = 0
                        matched_detections.add(min_track_id)
                    else:
                        new_track_id = len(all_tracks) + 1
                        make_new_track(new_track_id)
                        matched_detections.add(new_track_id)

        update_all_active_tracks()

        # Identify tracks to be removed from the active_tracks dictionary
        for track_id, track in active_tracks.items():
            if track.is_lost(max_frames_lost):
                tracks_to_remove.append(track_id)

        # Remove lost tracks from the active_tracks dictionary
        for track_id in tracks_to_remove:
            del active_tracks[track_id]

        # Print active track IDs for this frame
        print(f"Frame {frame}: Active Track IDs: {list(active_tracks.keys())}")

# Print the final list of all tracks
print("\nAll Tracks:")
for track in all_tracks:
    print(
        f"Track {track.track_id}: Trajectory Length: {len(track.previous_objects) + 1}\n Previous objects: {track.previous_objects} \n Current object: {track.current_object}")