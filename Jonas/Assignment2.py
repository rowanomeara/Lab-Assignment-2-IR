import numpy as np
from math import pi
from roboticstoolbox import DHRobot, DHLink
from spatialgeometry import Mesh
from spatialmath import SE3
import os


class DoosanM0609(DHRobot):
    def __init__(self):
        links = [
            DHLink(d=0.135,  a=0.0,   alpha=pi/2,  offset=0.0,    qlim=[-2*pi, 2*pi]),
            DHLink(d=0.0,    a=0.411, alpha=0.0,   offset=pi/2,   qlim=[-pi, pi]),
            DHLink(d=0.0,    a=0.0,   alpha=pi/2,  offset=pi/2,   qlim=[-2.618, 2.618]),
            DHLink(d=0.368,  a=0.0,   alpha=-pi/2, offset=0.0,    qlim=[-2*pi, 2*pi]),
            DHLink(d=0.0,    a=0.0,   alpha=pi/2,  offset=pi,     qlim=[-2*pi, 2*pi]),
            DHLink(d=0.121,  a=0.0,   alpha=0.0,   offset=0.0,    qlim=[-2*pi, 2*pi]),
        ]

        super().__init__(links, name="Doosan M0609")

        mesh_dir = os.path.join(os.path.dirname(__file__), "DoosanM0609Meshes")

        offsets = [
            np.array([
                [1.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  1.0,  0.0],
                [0.0, -1.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  1.0],
            ]),
            np.array([
                [1.0,  0.0,  0.0, -0.411],
                [0.0, -1.0,  0.0,  0.0],
                [0.0,  0.0, -1.0, -0.006],
                [0.0,  0.0,  0.0,  1.0],
            ]),
            np.array([
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0, -0.006],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ]),
            np.array([
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [ 0.0, -1.0,  0.0, -0.006],
                [ 0.0,  0.0,  0.0,  1.0],
            ]),
            np.array([
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0, -0.006],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ]),
            np.array([
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0, -0.006],
                [ 0.0,  0.0, -1.0, -0.242],
                [ 0.0,  0.0,  0.0,  1.0],
            ]),
        ]

        doosan_white = (0.92, 0.92, 0.92, 1.0)
        doosan_gray = (0.5, 0.5, 0.5, 1.0)

        def load_geom(filename, T_offset, color=doosan_white):
            mesh_path = os.path.join(mesh_dir, filename)
            if os.path.exists(mesh_path):
                m = Mesh(mesh_path)
                m.T = SE3(T_offset)
                m.color = color
                return m
            return None

        self.base_link_mesh = load_geom("base_link.stl", np.eye(4), color=doosan_gray)

        for i in range(6):
            m = load_geom(f"link_{i+1}.stl", offsets[i])
            self.links[i].geometry = [m] if m else []
            self.links[i].collision = [load_geom(f"link_{i+1}.stl", offsets[i])] if m else []

    def _update_link_tf(self, q=None):
        super(DHRobot, self)._update_link_tf(q)


if __name__ == "__main__":
    robot = DoosanM0609()
    print("DH Model Built")

    import swift
    env = swift.Swift()
    env.launch(realtime=True)

    if robot.base_link_mesh is not None:
        env.add(robot.base_link_mesh)

    env.add(robot)

    from roboticstoolbox import jtraj

    q_start = np.zeros(6)
    q_end = np.array([pi/4, -pi/6, pi/3, pi/4, pi/3, pi/6])

    traj = jtraj(q_start, q_end, 100)

    for q_step in traj.q:
        robot.q = q_step
        env.step(0.05)

    while True:
        env.step(0.05)