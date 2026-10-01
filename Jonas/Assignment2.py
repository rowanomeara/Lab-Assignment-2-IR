"""
Doosan Robotics M0609 6-DOF Collaborative Robot Arm.
UTS Industrial Robotics (Lab Assignment 2).
"""

import os
from math import pi
from typing import Optional, Union

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, jtraj
from spatialgeometry import Mesh
from spatialmath import SE3


class DoosanM0609(DHRobot):
    """Doosan Robotics M0609 6-DOF Collaborative Robot Arm."""

    def __init__(self, base: Optional[Union[SE3, np.ndarray]] = None, mesh_dir: Optional[str] = None):
        # Joint physical limits (radians)
        # Joints 1, 2, 4, 5, 6: +/- 360 deg (+/- 2*pi)
        # Joint 3: +/- 150 deg (+/- 2.618 rad)
        qlim_full = [-2 * pi, 2 * pi]
        qlim_j3 = [-2.618, 2.618]

        # Standard DH Parameters [d, a, alpha, offset]
        # d1 = 0.1525m (base to shoulder axis)
        # a2 = 0.411m (upper arm length)
        # d4 = 0.368m (forearm length)
        # d6 = 0.121m (wrist to tool flange)
        links = [
            RevoluteDH(d=0.1525, a=0.0,   alpha=pi / 2,  qlim=qlim_full),
            RevoluteDH(d=0.0,    a=0.411, alpha=0.0,     offset=pi / 2, qlim=[-pi, pi]),
            RevoluteDH(d=0.0,    a=0.0,   alpha=pi / 2,  offset=pi / 2, qlim=qlim_j3),
            RevoluteDH(d=0.368,  a=0.0,   alpha=-pi / 2, qlim=qlim_full),
            RevoluteDH(d=0.0,    a=0.0,   alpha=pi / 2,  offset=pi,     qlim=qlim_full),
            RevoluteDH(d=0.121,  a=0.0,   alpha=0.0,     qlim=qlim_full),
        ]

        super().__init__(links, name="Doosan M0609")

        if base is not None:
            if isinstance(base, np.ndarray):
                self.base = SE3(base)
            else:
                self.base = base

        # Locate mesh directory
        if mesh_dir is None:
            script_dir = os.path.abspath(os.path.dirname(__file__))
            candidate_dirs = [
                os.path.join(script_dir, "DoosanM0609Meshes"),
                os.path.join(script_dir, "DoosanM0609"),
                os.path.join(script_dir, "..", "DoosanM0609Meshes"),
                os.path.join(script_dir, "..", "DoosanM0609Links"),
                script_dir,
            ]
            for d in candidate_dirs:
                if os.path.exists(os.path.join(d, "base_link.stl")):
                    mesh_dir = os.path.abspath(d)
                    break
            if mesh_dir is None:
                mesh_dir = script_dir

        self.mesh_dir = mesh_dir

        # CAD-to-DH Offset Matrices (Offset = Inverse(DH_Frame) * CAD_Frame)
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

        # Colors: Doosan pearl white & dark gray
        doosan_white = (0.92, 0.92, 0.92, 1.0)
        doosan_gray = (0.3, 0.3, 0.3, 1.0)
        link_colors = [doosan_white, doosan_white, doosan_white, doosan_white, doosan_white, doosan_gray]

        def load_geom(filename, T_offset, color=doosan_white):
            path = os.path.join(mesh_dir, filename)
            if os.path.exists(path):
                m = Mesh(path, color=color)
                m.T = SE3(T_offset)
                return m
            return None

        self.base_link_mesh = load_geom("base_link.stl", np.eye(4), color=doosan_gray)

        for i in range(6):
            m = load_geom(f"link_{i+1}.stl", offsets[i], color=link_colors[i])
            self.links[i].geometry = [m] if m else []
            self.links[i].collision = [load_geom(f"link_{i+1}.stl", offsets[i], color=link_colors[i])] if m else []

    def _update_link_tf(self, q=None):
        super(DHRobot, self)._update_link_tf(q)

    def add_to_env(self, env):
        """Add both base link mesh and kinematic robot to Swift environment."""
        if self.base_link_mesh is not None:
            self.base_link_mesh.T = self.base
            env.add(self.base_link_mesh)
        env.add(self)

    def test(self, env=None):
        """Run a test motion trajectory in Swift and keep the window open."""
        created_env = False
        if env is None:
            import swift
            env = swift.Swift()
            env.launch(realtime=True)
            self.add_to_env(env)
            created_env = True

        q_start = self.q.copy()
        q_end = np.array([pi / 4, -pi / 6, pi / 3, pi / 4, pi / 3, pi / 6])
        traj = jtraj(q_start, q_end, 80)

        print("Doosan M0609 moving to target pose...")
        for q_step in traj.q:
            self.q = q_step
            env.step(0.04)

        print("Doosan M0609 motion test complete! Close browser window to exit.")
        while True:
            env.step(0.05)


if __name__ == "__main__":
    import swift

    env = swift.Swift()
    env.launch(realtime=True)

    robot = DoosanM0609(base=SE3(0, 0, 0))
    robot.add_to_env(env)

    # Initial render step so the robot appears immediately
    env.step(0)

    # Animate test trajectory and keep simulation running
    robot.test(env=env)