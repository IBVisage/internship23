import numpy as np


class Track:
    def __init__(self, tracks_id, initial_state):
        self.track_id = tracks_id
        self.current_object = initial_state
        self.previous_objects = []  # List to store previous object states
        self.frames_since_last_update = 0

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


# Initialize containers for tracks
active_tracks = {}  # Dictionary for active tracks
all_tracks = []  # List to keep track of all tracks

max_frames_lost = 3  # Maximum frames to consider a track lost

# Simulate adding new tracks and updating them over multiple frames
for frame in range(1, 10):
    # Simulate new object detections for this frame (replace this with your actual data)
    detections = [np.random.rand(5) for _ in range(np.random.randint(1, 5))]

    # List to keep track of tracks to be removed
    tracks_to_remove = []

    # Increment the age of all active tracks
    for track in active_tracks.values():
        track.increment_age()

    # Update existing tracks or create new ones by finding the closest track for each detection
    for detection in detections:
        min_distance = float("inf")  # Initialize with a large value
        closest_track_id = None

        for track_id, track in active_tracks.items():
            # Calculate the distance between the detection and the current object in the track
            distance = np.linalg.norm(detection - track.current_object)

            # Check if this track is closer than the previous closest track
            if distance < min_distance:
                min_distance = distance
                closest_track_id = track_id

        if closest_track_id is not None and min_distance < 0.7:  # Only update if within a threshold distance
            active_tracks[closest_track_id].update(detection)
        else:
            # Create a new track
            new_track_id = len(all_tracks) + 1
            new_track = Track(new_track_id, detection)
            active_tracks[new_track_id] = new_track
            all_tracks.append(new_track)

    # Identify tracks to be removed from the active_tracks dictionary
    for track_id, track in active_tracks.items():
        if track.is_lost(max_frames_lost):
            tracks_to_remove.append(track_id)

    # Remove lost tracks from the active_tracks dictionary
    for track_id in tracks_to_remove:
        del active_tracks[track_id]

    # Print active track IDs for this frame
    print(f"Detections are : {detections}")
    active_track_ids = list(active_tracks.keys())
    print(f"Frame {frame}: Active Track IDs: {active_track_ids}")

# Print the final list of all tracks
print("\nAll Tracks:")
for track in all_tracks:
    print(
        f"Track {track.track_id}: Trajectory Length: {len(track.previous_objects) + 1}\n Previous objects: {track.previous_objects} \n Current object: {track.current_object}")
