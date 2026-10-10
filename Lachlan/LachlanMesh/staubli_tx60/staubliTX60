import os
import sys
from math import pi
from typing import Optional, Union

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, jtraj
from spatialgeometry import Mesh
from spatialmath import SE3


class StaubliTX60(DHRobot):
    """Staubli TX60 6-DOF Industrial Robot Arm."""

    def __init__(self, base: Optional[Union[SE3, np.ndarray]] = None, mesh_dir: Optional[str] = None):
        qlim_full = [-2 * pi, 2 * pi]

        links = [
            RevoluteDH(d=0.375, a=0.0,   alpha=-pi / 2, qlim=qlim_full),
            RevoluteDH(d=0.0,   a=0.290, alpha=0.0,     offset=-pi / 2, qlim=qlim_full),
            RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  qlim=qlim_full),
            RevoluteDH(d=0.310, a=0.0,   alpha=-pi / 2, qlim=qlim_full),
            RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  qlim=qlim_full),
            RevoluteDH(d=0.070, a=0.0,   alpha=0.0,     qlim=qlim_full),
        ]

        super().__init__(links, name="Staubli TX60")

        if base is not None:
            self.base = SE3(base) if isinstance(base, np.ndarray) else base

        # --- FIX 1: Robust Absolute Path Resolution for OneDrive ---
        if mesh_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_dirs = [
                script_dir,
                os.path.join(script_dir, "LachlanMesh", "staubli_tx60"),
                os.path.join(script_dir, "..", "LachlanMesh", "staubli_tx60"),
            ]
            for d in candidate_dirs:
                if os.path.isfile(os.path.normpath(os.path.join(d, "link_1.stl"))):
                    mesh_dir = d
                    break
            if mesh_dir is None:
                mesh_dir = script_dir

        self.mesh_dir = os.path.normpath(os.path.abspath(mesh_dir))

        offsets = [
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ]

        staubli_yellow = (1.0, 0.7, 0.0, 1.0)
        staubli_gray = (0.3, 0.3, 0.3, 1.0)

        # --- FIX 2: Strict File Verification before loading Mesh ---
        for i in range(6):
            stl_path = os.path.normpath(os.path.join(self.mesh_dir, f"link_{i+1}.stl"))
            
            if os.path.isfile(stl_path):
                color = staubli_gray if i == 5 else staubli_yellow
                try:
                    m = Mesh(filename=stl_path, color=color)
                    m.T = SE3(offsets[i])
                    self.links[i].geometry = [m]
                except Exception as e:
                    print(f"Error loading mesh {stl_path}: {e}")
            else:
                print(f"Missing file: {stl_path}")


if __name__ == "__main__":
    import swift

    env = swift.Swift()
    env.launch(realtime=True)

    robot = StaubliTX60()
    env.add(robot)
    env.step(0)

    q_start = robot.q.copy()
    q_end = np.array([pi / 4, -pi / 6, pi / 3, pi / 4, pi / 3, pi / 6])
    traj = jtraj(q_start, q_end, 60)

    for q_step in traj.q:
        robot.q = q_step
        env.step(0.04)

    try:
        while True:
            env.step(0.05)
    except KeyboardInterrupt:
        print("Done.")