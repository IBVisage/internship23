

import shapely
import numpy as np
import math
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


rectangle1 = [1,1, 2, 2, 0]
rectangle2 = [1,1, 2, 4, 0]

corners1 = calculate_corners(rectangle1[0], rectangle1[1], rectangle1[2], rectangle1[3], rectangle1[4])
corners2 = calculate_corners(rectangle2[0], rectangle2[1], rectangle2[2], rectangle2[3], rectangle2[4])


poly1 = shapely.Polygon(corners1)
# polyt = shapely.Polygon(corners1)
poly2 = shapely.Polygon(corners2)

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




print(poly1.intersection(poly2).area)
print(poly1.union(poly2).area)

iou = poly1.intersection(poly2).area / poly1.union(poly2).area

plt.show()


pass



