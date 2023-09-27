import math
import numpy as np

def cart_2_polar(detections):
    pol_detections = []
    for detection in detections:
        if len(detection) == 0:
            return
        
        pol_indices = [1,2]
        
        tmp = [detection[i] for i in pol_indices] # Get 2 arguments
        # Calculate paramaters
        r = math.sqrt(float(tmp[0])**2  +  float(tmp[1])**2)
        theta = math.atan2(float(tmp[1]), float(tmp[0]))

        


        # Add them in form of a string
        detection = np.append(detection, r)
        detection = np.append(detection, theta)

        pol_detections.append(detection)
    
    return np.array(pol_detections)
