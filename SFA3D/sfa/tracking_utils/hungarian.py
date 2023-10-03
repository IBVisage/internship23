"""
# Authors: Ivan Bukač, Ante Ćubela
# DoC: 2023.10.06.
-----------------------------------------------------------------------------------
# Description:  Implementation of Hungarian-auction algorithm for track association in 2D object tracking problem
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_auction(detections, tracks, cost_func, threshold, thresh_ignore, frame):
    best_assignments = []
    unassigned_tracks = []
    unassigned_detections = []

    if detections.ndim == 1:
        detections = np.array([detections])

    row_index_to_track = {i: track for i, track in enumerate(tracks)}
    col_index_to_detection = {i: detection for i, detection in enumerate(detections)}

    num_detections = len(detections)
    num_tracks = len(tracks)
    cost_matrix = np.zeros((num_tracks, num_detections))

    if len(detections[0,:]):
        for track_idx, track in enumerate(tracks):
            for det_idx, detection in enumerate(detections):

                distance_comparison_element = tracks.get(track).current_object

                distance_comparison_element[1] = tracks.get(track).current_prediction[0]
                distance_comparison_element[2] = tracks.get(track).current_prediction[3]

                distance = cost_func(detection, distance_comparison_element, threshold)
                
                # Ako se koristi Euklidska udaljenost, potrebno je ovo dolje i thresh_ignore = 999

                # distance = cost_func(detection, distance_comparison_element)
                # if distance > threshold :
                #         distance = thresh_ignore

                cost_matrix[track_idx,det_idx] = distance

        min_better = False
        if min_better:
            cost_matrix = cost_matrix * (-1) + 1

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        best_assignments = [(row_index_to_track[row_idx], col_index_to_detection[col_idx]) for 
                            row_idx, col_idx in zip(row_indices, col_indices) if not int(cost_matrix[row_idx, col_idx]) == thresh_ignore]

    for track in tracks:
        assigned = False
        for pair in best_assignments:
            if track == pair[0]:
                assigned = True
                break
        if not assigned:
            unassigned_tracks.append(track)

    for detection in detections:
        assigned = False
        for pair in best_assignments:
            if np.array_equal(detection, pair[1]):
                assigned = True
                break
        if not assigned:
            unassigned_detections.append(detection)

    return best_assignments, unassigned_tracks, unassigned_detections


     




























# def calculate_cost_matrix(detections, tracks, cost_func, threshold, thresh_ignore):

    

#     d_n = len(detections)
#     t_n = len(tracks)
#     tracks = list(tracks)

#     # list of 

#     cost_matrix = np.zeros((t_n, d_n))
#     for ii in range(t_n):
#         for jj in range(d_n):
#                 # distance_comparison_element = tracks[ii].current_object
#                 # # distance_comparison_element[1] = act_track.current_prediction[0]
#                 # # distance_comparison_element[2] = act_track.current_prediction[3]
#                 # # distance = cosine_distance(detection, distance_comparison_element)
#                 # distance = cost_func(detections[jj], distance_comparison_element)


#                 distance_comparison_element = tracks[ii].current_object
#                 distance_comparison_element[1] = tracks[ii].current_prediction[0]
#                 distance_comparison_element[2] = tracks[ii].current_prediction[3]

#                 distance = cost_func(detections[jj], distance_comparison_element)

#                 # if too big then ignore it
#                 if distance > threshold :
#                      distance = thresh_ignore

#                 cost_matrix[ii,jj] = distance

    
                
#     return cost_matrix

# def hungarian_assigement(cost_matrix):
     
#     best_track_ind, best_detection_ind = linear_sum_assignment(cost_matrix)

#     best_assignments = list(zip(best_track_ind, best_detection_ind))

#     return best_assignments

# def create_combination_lists(best_assignments, tracks, detections):
     
#     num_tracks = len(tracks)
#     num_detections = len(detections)

#     used_tracks = []
#     used_detections = []
#     used_track_detection_pairs = []

#     # Create lists of used tracks and detections
#     for track_idx, detection_idx in best_assignments:
#         used_tracks.append(tracks[track_idx])
#         used_detections.append(detections[detection_idx])

#         used_track_detection_pairs.append((tracks[track_idx], detections[detection_idx]))


#     # Create lists of unused tracks and detections
#     unused_tracks = [track for track in tracks if track not in used_tracks]
#     unused_detections = [detection for detection in detections if detection not in used_detections]

#     # Now, used_tracks and used_detections contain the paired track-detection combinations,
#     # while unused_tracks and unused_detections contain the unused tracks and detections.

#     return used_track_detection_pairs, unused_tracks, unused_detections


# def hungarian(detections, tracks, cost_func, threshold, thresh_ignore):
     
     
     
#      cost_matrix = calculate_cost_matrix(detections, tracks, cost_func, threshold, thresh_ignore)

#      best_assignments = hungarian_assigement(cost_matrix)

#      used_track_detection_pairs, unused_tracks, unused_detections = create_combination_lists(best_assignments, tracks, detections)


#      return used_track_detection_pairs, unused_tracks, unused_detections


     
