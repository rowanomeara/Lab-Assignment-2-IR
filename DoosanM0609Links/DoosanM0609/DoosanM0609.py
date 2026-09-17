"""Doosan Robotics M0609 6-DOF Cobot Model for Swift Simulation.

Kinematics derived from official Doosan Robotics ROS description:
- Payload: 6 kg
- Reach: 900 mm (a2 = 411 mm, d4 = 368 mm, d6 = 121 mm)
- 6 Revolute Joints
"""

import os
from math import pi
from typing import Optional, Union

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class DoosanM0609(UTSMeshRobot):
    """Doosan Robotics M0609 6-DOF Collaborative Robot Arm."""

    manufacturer_url = "https://www.doosanrobotics.com/en/products/m-series/m0609"

    def __init__(
        self,
        base: Optional[Union[SE3, np.ndarray]] = None,
        mesh_dir: Optional[str] = None,
    ):
        # Joint physical limits from Doosan specifications (radians)
        # Joints 1, 2, 4, 5, 6: +/- 360 deg (+/- 2*pi)
        # Joint 3: +/- 150 deg (+/- 2.618 rad)
        qlim_full = [-2 * pi, 2 * pi]
        qlim_j3 = [-2.618, 2.618]

        # Standard DH Parameters [d, a, alpha, offset]
        links = [
            rtb.RevoluteDH(d=0.135, a=0.0,   alpha=pi / 2,  qlim=qlim_full),
            rtb.RevoluteDH(d=0.0,   a=0.411, alpha=0.0,     offset=pi / 2, qlim=qlim_full),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi / 2, qlim=qlim_j3),
            rtb.RevoluteDH(d=0.368, a=0.0,   alpha=-pi / 2, qlim=qlim_full),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi,     qlim=qlim_full),
            rtb.RevoluteDH(d=0.121, a=0.0,   alpha=0.0,     qlim=qlim_full),
        ]

        # Home / ready configuration
        home_q = [0.0, 0.0, pi / 2, 0.0, pi / 2, 0.0]

        if mesh_dir is None:
            script_dir = os.path.abspath(os.path.dirname(__file__))
            repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            candidate_dirs = [
                script_dir,
                os.path.join(script_dir, ".."),
                os.path.join(repo_root, "DoosanM0609Links"),
                os.path.join(script_dir, "DoosanM0609"),
            ]
            for d in candidate_dirs:
                if os.path.exists(os.path.join(d, "DoosanM0609Link0.dae")):
                    mesh_dir = os.path.abspath(d)
                    break

            if mesh_dir is None:
                for root, _, files in os.walk(repo_root):
                    if "DoosanM0609Link0.dae" in files:
                        mesh_dir = root
                        break

            if mesh_dir is None:
                mesh_dir = script_dir

        super().__init__(
            links=links,
            mesh_stem="DoosanM0609",
            mesh_dir=mesh_dir,
            name="DoosanM0609",
            home_q=home_q,
            base=base,
        )


if __name__ == "__main__":
    import swift

    env = swift.Swift()
    env.launch(realtime=True)

    robot = DoosanM0609(base=SE3(0, 0, 0))
    robot.add_to_env(env)
    print("Doosan M0609 added to Swift successfully!")
    robot.test()
