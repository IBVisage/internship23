import numpy as np
import tracking_utils.metric_utils as metric

from scipy.optimize import linear_sum_assignment



def testing_function(detections, tracks, cost_func, threshold, thresh_ignore, frame):
    best_assignments = []
    unassigned_tracks = []
    unassigned_detections = []

    if detections.ndim == 1:
         detections = np.array([detections])

    # print("DETECTIONS")
    # print(detections)
     
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

                distance = np.linalg.norm(detection[1:3] - distance_comparison_element[1:3])

                # if too big then ignore it
                if distance > threshold :
                        distance = thresh_ignore

                cost_matrix[track_idx,det_idx] = distance

    
        # Use the Hungarian algorithm to find the best assignments
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Map the indices back to the specific detections and tracks
        #best_assignments = [(row_index_to_track[row_idx], col_index_to_detection[col_idx]) for row_idx, col_idx in zip(row_indices, col_indices)]


        # stavlja stvari koje su očito predaleko u rješenje i takve treba ukloniti je rnikad se ne desi da postoji traka i detekcija u isto vrijeme nego ih namjerno doda nekoj i onda se gube
        # No ako se stavi uvjet da ne stavlja ove od 999 onda se niti jedna ne stavi što nema smisla
        best_assignments = [(row_index_to_track[row_idx], col_index_to_detection[col_idx]) for row_idx, col_idx in zip(row_indices, col_indices) if not int(cost_matrix[row_idx, col_idx]) == 999]


        # for row_idx, col_idx in zip(row_indices, col_indices):
        #     track = row_index_to_track[row_idx]
        #     detection = col_index_to_detection[col_idx]
        #     cost = cost_matrix[row_idx, col_idx]

        #     if not cost < threshold:
        #         best_assignments.append((track, detection))


        #print(cost_matrix)

   
    g = 0


    # Initialize lists to store tracks and detections that haven't been assigned
    
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
    
    g = 0
    pass
    
    # print("Detekcije bez para : " + str(len(unassigned_detections)))
    # print("Trake bez detekcija : " + str(len(unassigned_tracks)))

    # pokušaj si ispsivati z asvaki krug koje si poveznice sa kojim track id-ovim napravio i ako ti rade neke onda je vjeorjatno dobro
    # usporedi slike sa onime što se ispisuje, i provejri da li možeš dobro vratiti podatke nazad d ase mogu iskoristiti kako spada
    


    return best_assignments, unassigned_tracks, unassigned_detections
    # Now, best_assignments contains pairs of (track, detection) for the best assignments,
    # unassigned_tracks contains tracks that haven't been assigned, and
    # unassigned_detections contains detections that haven't been assigned.

     




























def calculate_cost_matrix(detections, tracks, cost_func, threshold, thresh_ignore):

    

    d_n = len(detections)
    t_n = len(tracks)
    tracks = list(tracks)

    # list of 

    cost_matrix = np.zeros((t_n, d_n))
    for ii in range(t_n):
        for jj in range(d_n):
                # distance_comparison_element = tracks[ii].current_object
                # # distance_comparison_element[1] = act_track.current_prediction[0]
                # # distance_comparison_element[2] = act_track.current_prediction[3]
                # # distance = cosine_distance(detection, distance_comparison_element)
                # distance = cost_func(detections[jj], distance_comparison_element)


                distance_comparison_element = tracks[ii].current_object
                distance_comparison_element[1] = tracks[ii].current_prediction[0]
                distance_comparison_element[2] = tracks[ii].current_prediction[3]

                distance = cost_func(detections[jj], distance_comparison_element)

                # if too big then ignore it
                if distance > threshold :
                     distance = thresh_ignore

                cost_matrix[ii,jj] = distance

    
                
    return cost_matrix

def hungarian_assigement(cost_matrix):
     
    best_track_ind, best_detection_ind = linear_sum_assignment(cost_matrix)

    best_assignments = list(zip(best_track_ind, best_detection_ind))

    return best_assignments

def create_combination_lists(best_assignments, tracks, detections):
     
    num_tracks = len(tracks)
    num_detections = len(detections)

    used_tracks = []
    used_detections = []
    used_track_detection_pairs = []

    # Create lists of used tracks and detections
    for track_idx, detection_idx in best_assignments:
        used_tracks.append(tracks[track_idx])
        used_detections.append(detections[detection_idx])

        used_track_detection_pairs.append((tracks[track_idx], detections[detection_idx]))


    # Create lists of unused tracks and detections
    unused_tracks = [track for track in tracks if track not in used_tracks]
    unused_detections = [detection for detection in detections if detection not in used_detections]

    # Now, used_tracks and used_detections contain the paired track-detection combinations,
    # while unused_tracks and unused_detections contain the unused tracks and detections.

    return used_track_detection_pairs, unused_tracks, unused_detections


def hungarian(detections, tracks, cost_func, threshold, thresh_ignore):
     
     
     
     cost_matrix = calculate_cost_matrix(detections, tracks, cost_func, threshold, thresh_ignore)

     best_assignments = hungarian_assigement(cost_matrix)

     used_track_detection_pairs, unused_tracks, unused_detections = create_combination_lists(best_assignments, tracks, detections)


     return used_track_detection_pairs, unused_tracks, unused_detections


     
