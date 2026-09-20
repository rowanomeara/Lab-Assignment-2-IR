import numpy as np
from math import pi
from spatialmath import SE3
from roboticstoolbox import DHRobot, DHLink

"""
6DOF Offset Calculator
-----------------------------
Yo Jonas and Lachlan...
I made this calculator to derive the offset matrices to map 
the CAD meshes onto DH coordinate frames
Here is how you use it...

1. Get the DH parameters for your robot from a textbook or something
2. Stack SE3 matrix distances to build your physical robot (From the manufacturer specsheet/URDF)
3. Let the calc find the offset (offset = Inverse(DH_Frame) * CAD_Frame)
"""

# DH PARAMETER ROBOT
# Add in your DH parameters
dh_links = [
    DHLink(d=0.290, a=0.0,   alpha=-pi/2, offset=0.0),
    DHLink(d=0.0,   a=0.270, alpha=0.0,   offset=-pi/2),
    DHLink(d=0.0,   a=0.070, alpha=-pi/2, offset=0.0),
    DHLink(d=0.302, a=0.0,   alpha=pi/2,  offset=0.0),
    DHLink(d=0.0,   a=0.0,   alpha=-pi/2, offset=0.0),
    DHLink(d=0.072, a=0.0,   alpha=0.0,   offset=0.0)
]
dh_robot = DHRobot(dh_links)


# DEFINE THE TRUE PHYSICAL LOCATIONS 
# Add in the physical measurements written in the manufacturer's manual for the robot you chose 

T_cad = [
    # Link 1: Shoulder is 0.29m straight up from the floor
    SE3(0, 0, 0.29),                                       
    
    # Link 2: Elbow is another 0.27m straight up from the shoulder
    SE3(0, 0, 0.29) * SE3(0, 0, 0.27),                     
    
    # Link 3: Forearm shifts slightly up and 0.302m forward
    SE3(0, 0, 0.29) * SE3(0, 0, 0.27) * SE3(0.302, 0, 0.07), 
    
    # Link 4 & 5: For my robot the wrist motors intersect at the same point as Link 3
    SE3(0, 0, 0.29) * SE3(0, 0, 0.27) * SE3(0.302, 0, 0.07), 
    SE3(0, 0, 0.29) * SE3(0, 0, 0.27) * SE3(0.302, 0, 0.07), 
    
    # Link 6: The final link is 0.072m out from the wrist
    SE3(0, 0, 0.29) * SE3(0, 0, 0.27) * SE3(0.302, 0, 0.07) * SE3(0, 0, 0.072) 
]


# CALCULATE THE OFFSETS

T_dh_global = SE3()
for i in range(6):
    
    T_dh_global = T_dh_global * dh_robot.links[i].A(0)
    
    # Offset = Inverse(DH_Frame) * CAD_Frame)
    T_offset = T_dh_global.inv() * T_cad[i]
    
    print(f"OFFSET FOR LINK {i+1}:")
    print("np.array([")
    for row in T_offset.A:
        row_clean = [round(val, 5) if abs(val) > 1e-10 else 0.0 for val in row]
        print(f"    {row_clean},")
    print("]),\n")