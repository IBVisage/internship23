import os
import numpy as np

def read_data_from_txt(path_to_file):
    array_2d_list = []
    array_3d_list = []

    current_array = None

    with open(path_to_file, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:
                current_array = None
                continue

            values = [float(value) for value in line.split('\t')]

            if current_array is None:
                if len(values) == 8:
                    current_array = array_2d_list
                else:
                    current_array = []  # Initialize a temporary list for 3D data

            current_array.append(values)

            if len(current_array) == 8:
                array_3d_list.append(current_array)  # Append the 3D data as a whole

    array_2d = np.array(array_2d_list)
    array_3d = np.array(array_3d_list)

    return array_2d, array_3d


video_number = input("Input the video number (00 - 28) : ")

# Set the current directory as the working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory + "/outputs/outputs_video_" + video_number)

# Define the file name (assuming it's in the current directory)
file_name = 'out_000000.txt'  # Replace with the actual file name
array_2d, array_3d = read_data_from_txt(file_name)
print("2D Array:")
print(array_2d)
print("3D Array:")
print(array_3d)

print(array_2d.shape)
print(array_3d.shape)
