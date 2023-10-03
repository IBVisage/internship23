import numpy as np
import matplotlib.pyplot as plt
import math


def plot_objects(object_data, ids_data, orientation_data, ax):
    ax.clear()

    for obj_id, obj in enumerate(object_data):
        x = obj[0]
        y = obj[3]
        width = orientation_data[obj_id][0]
        length = orientation_data[obj_id][1]
        orientation = orientation_data[obj_id][2]
        identification = ids_data[obj_id]

        x1 = x - length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y1 = y - length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x2 = x + length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y2 = y + length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x3 = x + length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y3 = y + length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)
        x4 = x - length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y4 = y - length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)

        front_edge = [[x1, y1], [x2, y2]]
        left_edge = [[x1, y1], [x4, y4]]
        right_edge = [[x2, y2], [x3, y3]]
        back_edge = [[x3, y3], [x4, y4]]

        ax.plot(*zip(*front_edge), color='red', linewidth=2)
        ax.plot(*zip(*left_edge), color='red', linewidth=2)
        ax.plot(*zip(*right_edge), color='lightblue', linewidth=2)
        ax.plot(*zip(*back_edge), color='red', linewidth=2)

        obj_center_x = (x1 + x3) / 2
        obj_center_y = (y1 + y3) / 2
        ax.text(obj_center_x, obj_center_y, str(identification), color='black', ha='center', va='center', fontsize=12)

    ax.set_xlim(0, 50)
    ax.set_ylim(25, -25)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.gca().invert_yaxis()
    plt.draw()
    plt.pause(0.25)
