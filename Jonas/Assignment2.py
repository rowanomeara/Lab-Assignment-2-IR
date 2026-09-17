import os
from math import pi
import roboticstoolbox as rtb
from spatialmath import SE3
from ir_support.robots.UTSMeshRobot import UTSMeshRobot

class DoosanM0609(UTSMeshRobot):
    """Doosan Robotics M0609 6-DOF Collaborative Robot Arm."""

    def __init__(self, base=None):
        # Define the standard DH parameters [d, a, alpha, offset] and joint limits [qlim]
        links = [
            rtb.RevoluteDH(d=0.135, a=0.0,   alpha=pi / 2,  qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.0,   a=0.411, alpha=0.0,     offset=pi / 2, qlim=[-pi, pi]),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi / 2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.368, a=0.0,   alpha=-pi / 2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi,     qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.121, a=0.0,   alpha=0.0,     qlim=[-2 * pi, 2 * pi]),
        ]

        mesh_dir = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(mesh_dir, "DoosanM0609Link0.dae")):
            mesh_dir = os.path.join(mesh_dir, "DoosanM0609")

        super().__init__(
            links=links,
            mesh_stem="DoosanM0609",
            mesh_dir=mesh_dir,
            name="DoosanM0609",
            home_q=[0.0, 0.0, pi / 2, 0.0, pi / 2, 0.0],
            base=base,
        )


import swift
from spatialmath import SE3

env = swift.Swift()
env.launch(realtime=True)

# Instantiate and add robot to Swift environment
robot = DoosanM0609(base=SE3(0, 0, 0))
robot.add_to_env(env)

# Verify motion
robot.test()