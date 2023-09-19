import shapely
import numpy as np
import matplotlib.pyplot as plt

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
    R = rotate(orientation)
    w, l = width, length
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners], dtype=np.float32)
    corners_2d = np.dot(R, corners)
    corners_2d = corners_2d + np.array((center_x, center_y), dtype=np.float32).reshape(2, 1)
    
    # Return x and y pairs
    return [(corners_2d[0,0], corners_2d[1,0]),
            (corners_2d[0,1], corners_2d[1,1]),
            (corners_2d[0,2], corners_2d[1,2]),
            (corners_2d[0,3], corners_2d[1,3])]

def visualise(poly1, poly2):
        # Visualisation
    x1, y1 = poly1.exterior.xy
    x2, y2 = poly2.exterior.xy

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # Plot the polygons
    ax.plot(x1, y1, label='Polygon 1', color='blue')
    ax.fill(x1, y1, alpha=0.2, color='blue')  # Fill the polygon area

    ax.plot(x2, y2, label='Polygon 2', color='red')
    ax.fill(x2, y2, alpha=0.2, color='red')  # Fill the polygon area

    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()

    plt.show()

def intersection_over_union(object1, object2):
    # x->1, y->2, 5->w, 6->l, 7->phi
    index = [1, 2, 5, 6, 7]

    x1, y1, w1, l1, phi1 = [object1[ii] for ii in index]
    x2, y2, w2, l2, phi2 = [object2[ii] for ii in index]

    corners1 = calculate_corners(x1, y1, w1, l1, phi1)
    corners2 = calculate_corners(x2, y2, w2, l2, phi2)

    poly1 = shapely.Polygon(corners1)
    poly2 = shapely.Polygon(corners2)

    #visualise(poly1, poly2)
    
    iou = 0
    if poly1.intersects(poly2):
        iou = poly1.intersection(poly2).area / poly1.union(poly2).area

    return iou


# car1 = [1.00000000,	19.23171345,	16.39022325,	-1.87868116,	1.57203937,	1.66135744,	4.50914534,
#         	-1.59797299,	25.268522323391835,	0.8649976077597357]

# car2 = [1.00000000,	19.44323464,	-0.30678950,	-1.54884861,	1.61409998,	1.82322875,	4.36537266,
#          -1.58108211,	19.445654863321167,	1.5865737449664037]

# car1 = [0, 3, 15, 0, 0, 2, 4, 0.45, 0, 0]
# car2 = [0, 3, 0, 0, 0, 2, 4, 0, 0, 0]

iou = intersection_over_union(car1, car2)

print(iou)
