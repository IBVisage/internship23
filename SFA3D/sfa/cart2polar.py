import math
import os

def process_file(file_path, out_path, pol_indices = [1, 2]):
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            write_line = ""
            write_line_list = []
            

            with open(out_path, 'w') as out_file:

                for line in lines:
                    # Split the line into numbers
                    numbers = list(line.strip().split('\t'))

                    if len(numbers) > 2:
                        # Create polar coordinates from the first 2 arguments and replace them
                        tmp = [numbers[i] for i in pol_indices] # Get 2 arguments

                        # Calculate paramaters
                        r = math.sqrt(float(tmp[0])**2  +  float(tmp[1])**2)
                        theta = math.atan2(float(tmp[1]), float(tmp[0]))


                        # Add them in form of a string
                        numbers.append(str(r))
                        numbers.append(str(theta))

                        write_line = "\t".join(numbers) + "\n"
                        
                        write_line_list.append(write_line)

                out_file.writelines(write_line_list)
            

    except FileNotFoundError:
        return None

def process_folder(read_folder, write_folder):
    read_f_names = os.listdir(read_folder)
    
    # Create needed directory if there isn't any
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    for file in read_f_names:
        process_file(read_folder+ "/" + file, write_folder+"/"+file)


cart_folder_path = "outputs/out_vid_all/out_vid_"
polar_folder_path = "outputs/out_vid_polar_all/out_vid_"

folder_nums = [f"{num:02d}" for num in range(28 + 1)]

for ii in folder_nums:
    process_folder(cart_folder_path + ii, polar_folder_path + ii)


