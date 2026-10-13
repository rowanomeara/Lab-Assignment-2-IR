import numpy as np
from math import pi
from roboticstoolbox import DHRobot, DHLink
from spatialgeometry import Mesh
from spatialmath import SE3
import os


class StaubliTX60(DHRobot):
    def __init__(self):
        links = [
            DHLink(d=0.375, a=0.0,   alpha=-pi/2, offset=0.0,   qlim=[-2*pi, 2*pi]),
            DHLink(d=0.0,   a=0.290, alpha=0.0,   offset=-pi/2, qlim=[-2*pi, 2*pi]),
            DHLink(d=0.020, a=0.0,   alpha=pi/2,  offset=pi/2,  qlim=[-2*pi, 2*pi]),
            DHLink(d=0.310, a=0.0,   alpha=-pi/2, offset=0.0,   qlim=[-2*pi, 2*pi]),
            DHLink(d=0.0,   a=0.0,   alpha=pi/2,  offset=0.0,   qlim=[-2*pi, 2*pi]),
            DHLink(d=0.070, a=0.0,   alpha=0.0,   offset=0.0,   qlim=[-2*pi, 2*pi]),
        ]

        super().__init__(links, name="Staubli TX60")

        mesh_dir = os.path.dirname(os.path.abspath(__file__))

        offsets = [
            np.array([
                [1.0, 0.0,  0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0,  0.0, 0.0],
                [0.0, 0.0,  0.0, 1.0],
            ]),
            np.array([
                [0.0, 0.0, 1.0, -0.29],
                [1.0, 0.0, 0.0,  0.0],
                [0.0, 1.0, 0.0,  0.0],
                [0.0, 0.0, 0.0,  1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0,  0.0, 0.0],
                [0.0, 0.0, -1.0, 0.31],
                [0.0, 1.0,  0.0, 0.0],
                [0.0, 0.0,  0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ]

        staubli_yellow = (1.0, 0.7, 0.0, 1.0)
        staubli_gray = (0.3, 0.3, 0.3, 1.0)

        def load_geom(filename, T_offset, color=staubli_yellow):
            mesh_path = os.path.join(mesh_dir, filename)
            if os.path.exists(mesh_path):
                m = Mesh(mesh_path)
                m.T = SE3(T_offset)
                m.color = color
                return m
            return None

        self.base_link_mesh = load_geom("base_link.stl", np.eye(4), color=staubli_yellow)

        for i in range(6):
            c = staubli_gray if i == 5 else staubli_yellow
            m = load_geom(f"link_{i+1}.stl", offsets[i], color=c)
            self.links[i].geometry = [m] if m else []
            self.links[i].collision = [load_geom(f"link_{i+1}.stl", offsets[i], color=c)] if m else []

    def _update_link_tf(self, q=None):
        super(DHRobot, self)._update_link_tf(q)


if __name__ == "__main__":
    robot = StaubliTX60()

    import swift
    env = swift.Swift()
    env.launch(realtime=True)

    if robot.base_link_mesh is not None:
        env.add(robot.base_link_mesh)

    env.add(robot)

    from roboticstoolbox import jtraj

    q_start = np.zeros(6)
    q_end = np.array([pi/4, -pi/6, pi/3, pi/4, pi/3, pi/6])

    traj = jtraj(q_start, q_end, 60)

    try:
        for q_step in traj.q:
            robot.q = q_step
            env.step(0.04)

        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        pass