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

antigravity made ts im crine bruh

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

def calculate_offsets(robot_name, dh_links, T_cad):
    print(f"========================================")
    print(f" OFFSETS FOR {robot_name}")
    print(f"========================================")
    robot = DHRobot(dh_links)
    T_dh_global = SE3()
    for i in range(len(dh_links)):
        T_dh_global = T_dh_global * robot.links[i].A(0)
        # Offset = Inverse(DH_Frame) * CAD_Frame
        T_offset = T_dh_global.inv() * T_cad[i]
        print(f"OFFSET FOR LINK {i+1}:")
        print("np.array([")
        for row in T_offset.A:
            row_clean = [round(val, 5) if abs(val) > 1e-10 else 0.0 for val in row]
            print(f"    {row_clean},")
        print("]),\n")

if __name__ == "__main__":
    # Rowan's ABB IRB 120
    calculate_offsets("ABB IRB 120", dh_links, T_cad)

    # Jonas's Doosan M0609
    doosan_dh = [
        DHLink(d=0.1525, a=0.0,   alpha=pi/2,  offset=0.0),
        DHLink(d=0.0,    a=0.411, alpha=0.0,   offset=pi/2),
        DHLink(d=0.0,    a=0.0,   alpha=pi/2,  offset=pi/2),
        DHLink(d=0.368,  a=0.0,   alpha=-pi/2, offset=0.0),
        DHLink(d=0.0,    a=0.0,   alpha=pi/2,  offset=pi),
        DHLink(d=0.121,  a=0.0,   alpha=0.0,   offset=0.0)
    ]
    T0 = SE3()
    T1 = T0 * SE3(0, 0, 0.1525)
    T2 = T1 * SE3(0, 0.006, 0) * SE3.Rz(-pi/2) * SE3.Ry(-pi/2)
    T3 = T2 * SE3(0.411, 0, 0) * SE3.Rz(pi/2)
    T4 = T3 * SE3(0, -0.368, 0) * SE3.Rx(pi/2)
    T5 = T4 * SE3(0, 0, 0) * SE3.Rx(-pi/2)
    T6 = T5 * SE3(0, -0.121, 0) * SE3.Rx(pi/2)
    doosan_cad = [T1, T2, T3, T4, T5, T6]
    calculate_offsets("Doosan M0609", doosan_dh, doosan_cad)

    # Staubli TX60 RAAHHAHAHAHA
   
    staubli_dh = [
        DHLink(d=0.375, a=0.0,   alpha=-pi/2, offset=0.0),
        DHLink(d=0.0,   a=0.290, alpha=0.0,   offset=-pi/2),
        DHLink(d=0.0,   a=0.0,   alpha=pi/2,  offset=0.0),
        DHLink(d=0.310, a=0.0,   alpha=-pi/2, offset=0.0),
        DHLink(d=0.0,   a=0.0,   alpha=pi/2,  offset=0.0),
        DHLink(d=0.070, a=0.0,   alpha=0.0,   offset=0.0)
    ]
    
    
    T0 = SE3()
    T1 = T0 * SE3(0, 0, 0.375)                  
    T2 = T1 * SE3(0, 0, 0)                     
    T3 = T2 * SE3(0, 0.02, 0.290)              
    T4 = T3 * SE3(0, 0, 0)                      
    T5 = T4 * SE3(0, 0, 0.310)                  
    T6 = T5 * SE3(0, 0, 0.070)                  
    
    staubli_cad = [T1, T2, T3, T4, T5, T6]
    calculate_offsets("Staubli TX60", staubli_dh, staubli_cad)
