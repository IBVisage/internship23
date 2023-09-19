import shapely
import numpy as np
import matplotlib.pyplot as plt

def calculate_corners(center_x, center_y, width, length, orientation):
    half_width = width / 2
    half_length = length / 2

    # Calculate the coordinates of the four corners
    x1 = center_x + half_width * np.cos(orientation) - half_length * np.sin(orientation)
    y1 = center_y + half_width * np.sin(orientation) + half_length * np.cos(orientation)

    x2 = center_x - half_width * np.cos(orientation) - half_length * np.sin(orientation)
    y2 = center_y - half_width * np.sin(orientation) + half_length * np.cos(orientation)

    x3 = center_x + half_width * np.cos(orientation) + half_length * np.sin(orientation)
    y3 = center_y + half_width * np.sin(orientation) - half_length * np.cos(orientation)

    x4 = center_x - half_width * np.cos(orientation) + half_length * np.sin(orientation)
    y4 = center_y - half_width * np.sin(orientation) - half_length * np.cos(orientation)

    return [(x1, y1), (x2, y2), (x4, y4), (x3, y3)]

def intersection_over_union(object1, object2):
    # x->1, y->2, 5->w, 6->l, 7->phi
    index = [1, 2, 5, 6, 7]

    x1, y1, w1, l1, phi1 = [object1[ii] for ii in index]
    x2, y2, w2, l2, phi2 = [object2[ii] for ii in index]

    corners1 = calculate_corners(x1, y1, w1, l1, phi1)
    corners2 = calculate_corners(x2, y2, w2, l2, phi2)

    poly1 = shapely.Polygon(corners1)
    poly2 = shapely.Polygon(corners2)
    
    iou = 0
    if poly1.intersects(poly2):
        iou = poly1.intersection(poly2).area / poly1.union(poly2).area

    return iou


car1 = [1.00000000,	19.23171345,	16.39022325,	-1.87868116,	1.57203937,	1.66135744,	4.50914534,
        	-1.59797299,	25.268522323391835,	0.8649976077597357]

car2 = [1.00000000,	19.44323464,	-0.30678950,	-1.54884861,	1.61409998,	1.82322875,	4.36537266,
         -1.58108211,	19.445654863321167,	1.5865737449664037]

car1 = [0, 3, 20, 0, 0, 2, 3, 0.5, 0, 0]
car2 =[0, 3, 3, 0, 0, 2, 3, 0.5, 0, 0]

iou = intersection_over_union(car1, car2)

print(iou)
