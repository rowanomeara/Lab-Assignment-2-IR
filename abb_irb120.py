import numpy as np
from math import pi
from roboticstoolbox import DHRobot, DHLink, BaseRobot
from spatialgeometry import Mesh
from spatialmath import SE3
import os

class ABB_IRB120(DHRobot):
    def __init__(self):
        links = [
            DHLink(d=0.290, a=0.0,   alpha=-pi/2, offset=0.0,    qlim=[-165*pi/180, 165*pi/180]),
            DHLink(d=0.0,   a=0.270, alpha=0.0,   offset=-pi/2,  qlim=[-110*pi/180, 110*pi/180]),
            DHLink(d=0.0,   a=0.070, alpha=-pi/2, offset=0.0,    qlim=[-110*pi/180,  70*pi/180]),
            DHLink(d=0.302, a=0.0,   alpha=pi/2,  offset=0.0,    qlim=[-160*pi/180, 160*pi/180]),
            DHLink(d=0.0,   a=0.0,   alpha=-pi/2, offset=0.0,    qlim=[-120*pi/180, 120*pi/180]),
            DHLink(d=0.072, a=0.0,   alpha=0.0,   offset=0.0,    qlim=[-400*pi/180, 400*pi/180])
        ]
        
        super().__init__(links, name="ABB IRB 120 (DH)")
        
        mesh_dir = os.path.join(os.path.dirname(__file__), "IRB 120 Meshes")
        
        offsets = [
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.29],
                [0.0, 1.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, -0.0, 1.0, -0.27],
                [1.0, 0.0, -0.0, 0.0],
                [0.0, 1.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, -0.0, 1.0, -0.07],
                [0.0, -1.0, -0.0, -0.0],
                [1.0, 0.0, -0.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, -0.0, 1.0, 0.0],
                [1.0, 0.0, -0.0, -0.302],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, -0.0, 1.0, 0.0],
                [0.0, -1.0, -0.0, -0.0],
                [1.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [0.0, -0.0, 1.0, 0.0],
                [0.0, -1.0, -0.0, 0.0],
                [1.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        ]

        # orange colour
        abb_orange = (1.0, 0.45, 0.0, 1.0)

        def load_geom(filename, T_offset, color=abb_orange):
            mesh_path = os.path.join(mesh_dir, filename)
            if os.path.exists(mesh_path):
                m = Mesh(mesh_path)
                m.T = SE3(T_offset)
                m.color = color
                return m
            return None
                
        self.base_link_mesh = load_geom("base_link.stl", np.eye(4), color=(0.2, 0.2, 0.2, 1.0))
        
        for i in range(6):
            m = load_geom(f"link_{i+1}.stl", offsets[i])
            self.links[i].geometry = [m] if m else []
            self.links[i].collision = [load_geom(f"link_{i+1}.stl", offsets[i])] if m else []

    def _update_link_tf(self, q=None):
        super(DHRobot, self)._update_link_tf(q)

if __name__ == "__main__":
    robot = ABB_IRB120()
    print("Perfect Pure DH Model Built!")
    
    import swift
    env = swift.Swift()
    env.launch(realtime=True)
    
    if robot.base_link_mesh is not None:
        env.add(robot.base_link_mesh)
        
    env.add(robot)
    
    from roboticstoolbox import jtraj
    
    q_start = np.zeros(6)
    q_end = np.array([pi/4, pi/4, pi/4, pi/4, pi/4, pi/4])
    
    traj = jtraj(q_start, q_end, 100)
    
    for q_step in traj.q:
        robot.q = q_step
        env.step(0.05)
        
    while True:
        env.step(0.05)
