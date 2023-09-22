"""
Privremeni komadićak koda koji će sa danim vrhovima 3D boxa nacrtat vrhove toga boxa i stranice mu popunit, tj. obojat, za object tracking.
"""

import numpy as np
import cv2


def probavam_crtat(image_path, vertices_list):
    # Create an empty canvas with the specified dimensions
    image_width = 1242
    image_height = 375
    canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8) #canvas = ucitana slika
    # Sample vertices data
   # vertices_list = [
    #    [[717, 233], [775, 233], [788, 179], [707, 199], [717, 246], [778, 246], [778, 181], [717, 181]],
     #   [[701, 200], [671, 200], [671, 172], [701, 172], [722, 197], [694, 197], [694, 172], [722, 172]]
    #]
    # Define the color (BGR format)

    color = (0, 255, 0)  # Green color for edges and sides
    light_color = (128, 255, 128)  # Light green color for sides
    vertex_color = (0, 0, 255)  # Red color for vertices
    alpha = 0.4  # Transparency level (adjust as needed)

    # Define the indices for connecting vertices to form sides
    side_indices = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (0, 1, 2, 3), (4, 5, 6, 7)]

    # Draw edges, sides, and mark vertices
    for vertices in vertices_list:
        # Draw edges
        for i in range(4):
            cv2.line(canvas, tuple(vertices[i]), tuple(vertices[(i + 1) % 4]), color, 2)

        # Draw sides with light color
        for indices in side_indices:
            side_pts = [vertices[idx] for idx in indices]
            cv2.fillPoly(canvas, [np.array(side_pts, np.int32)], light_color)

        # Mark vertices with red circles
        for vertex in vertices:
            cv2.circle(canvas, tuple(vertex), 5, vertex_color, -1)

    # Overlay the canvas with transparency
    cv2.addWeighted(canvas, alpha, canvas.copy(), 1 - alpha, 0, canvas)

    # Display the canvas
    cv2.imshow('3D Objects', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canvas