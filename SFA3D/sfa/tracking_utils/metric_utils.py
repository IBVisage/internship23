import shapely
import numpy as np
import config.kitti_config as cnf
from copy import copy


def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 1.0

    return 1.0 - (dot_product / (norm_vec1 * norm_vec2))


def euclidian(vec1, vec2):

    return np.linalg.norm(vec1[1:3] - vec2[1:3])


def rotate(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s],
                     [s, c]])


def calculate_corners(center_x, center_y, width, length, orientation):
    # Calculate the coordinates of the four corners
    # dim: 3
    # location: 3
    # ry: 1
    # return: 8 x 3
    r = rotate(orientation)
    w, l = width, length
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners], dtype=np.float32)
    corners_2d = np.dot(r, corners)
    corners_2d = corners_2d + np.array((center_x, center_y), dtype=np.float32).reshape(2, 1)

    return [(corners_2d[0, 0], corners_2d[1, 0]),
            (corners_2d[0, 1], corners_2d[1, 1]),
            (corners_2d[0, 2], corners_2d[1, 2]),
            (corners_2d[0, 3], corners_2d[1, 3])]





def intersection_over_union(object1, object2):
    # x->1, y->2, 5->w, 6->l, 7->phi
    index = [1, 2, 5, 6, 7]

    x1, y1, w1, l1, phi1 = [object1[ii] for ii in index]
    x2, y2, w2, l2, phi2 = [object2[ii] for ii in index]

    corners1 = calculate_corners(x1, y1, 3 * w1, 3 * l1, phi1)
    corners2 = calculate_corners(x2, y2, 3 * w2, 3 * l2, phi2)

    poly1 = shapely.Polygon(corners1)
    poly2 = shapely.Polygon(corners2)

    iou = 0
    if poly1.intersects(poly2):
        iou = poly1.intersection(poly2).area / poly1.union(poly2).area

    return iou 


def k_iou_euc(object1, object2, euc_thr):
    ignore_cost=1
    # minimisation problem
    w_iou = 3 
    w_euc = 1

    iou = intersection_over_union(object1, object2)

    euc = euclidian(object1, object2)

    # Skip if too far away, we know that iou will be small
    if euc > euc_thr:
        return ignore_cost

    ## euclidean normalisation
    n_euc = (euc - 0) / (np.sqrt(cnf.boundary['maxX']**2 + (cnf.boundary['maxY'] - cnf.boundary['minY']-25)**2) - 0)


    total_weight = w_iou + w_euc
    w_iou /= total_weight
    w_euc /= total_weight

    
    combined_cost = (w_euc * n_euc) + (w_iou * (1 - iou))


    return combined_cost



