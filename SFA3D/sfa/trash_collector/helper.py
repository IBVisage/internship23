import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.backend_tools import ToolBase


# Initialize the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Initialize object data list with initial data as NumPy arrays
object_data_list = [
    np.array([[25, 0, 10, 4, 0]]),  # Example object 1
    np.array([[15, 10, 8, 6, math.pi / 4], [35, -10, 12, 3, -math.pi / 3]])  # Example objects 2 and 3
]

# Initialize object index and current object data
object_index = 0
current_object_data = object_data_list[object_index]


# Function to plot objects with specified parameters and IDs
def plot_objects(object_data):
    ax.clear()
    for obj_id, obj in enumerate(object_data):
        x, y, length, width, orientation = obj

        # Calculate the coordinates of the four corners of the box
        x1 = x - length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y1 = y - length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x2 = x + length / 2 * math.cos(orientation) + width / 2 * math.sin(orientation)
        y2 = y + length / 2 * math.sin(orientation) - width / 2 * math.cos(orientation)
        x3 = x + length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y3 = y + length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)
        x4 = x - length / 2 * math.cos(orientation) - width / 2 * math.sin(orientation)
        y4 = y - length / 2 * math.sin(orientation) + width / 2 * math.cos(orientation)

        # Create the box as a patch
        # box = patches.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, edgecolor='b', facecolor='none')

        # Define the vertices of each edge
        front_edge = [[x1, y1], [x2, y2]]
        left_edge = [[x1, y1], [x4, y4]]
        right_edge = [[x2, y2], [x3, y3]]
        back_edge = [[x3, y3], [x4, y4]]

        # Plot each edge with the specified colors
        ax.plot(*zip(*front_edge), color='lightblue', linewidth=2)
        ax.plot(*zip(*left_edge), color='red', linewidth=2)
        ax.plot(*zip(*right_edge), color='red', linewidth=2)
        ax.plot(*zip(*back_edge), color='red', linewidth=2)

        # Plot the box on the coordinate system
        # ax.add_patch(box)

        # Add the object ID as text near the object center
        obj_center_x = (x1 + x3) / 2
        obj_center_y = (y1 + y3) / 2
        ax.text(obj_center_x, obj_center_y, str(obj_id), color='black', ha='center', va='center', fontsize=12)

    # Set axis limits and labels
    ax.set_xlim(0, 50)
    ax.set_ylim(-25, 25)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Draw the plot
    plt.gca().invert_yaxis()
    plt.draw()


# Function to handle key press events
def on_key(event):
    global object_index, object_data_list, current_object_data

    if event.key == 'right':  # Right arrow key
        # Implement your algorithm to generate new object data here
        # For this example, we will just rotate the existing objects
        new_object_data = []
        for obj in current_object_data:
            x, y, length, width, orientation = obj
            orientation += math.pi / 6  # Rotate by 30 degrees (pi/6 radians)
            new_object_data.append(np.array([x, y, length, width, orientation]))
        object_data_list.append(new_object_data)

        # Update the object index and current object data
        object_index += 1
        current_object_data = object_data_list[object_index]

        # Plot the new objects
        plot_objects(current_object_data)


# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)

print(current_object_data)
# Plot the initial objects
plot_objects(current_object_data)

# Show the plot
plt.show()
