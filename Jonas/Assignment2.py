import os
from math import pi
import roboticstoolbox as rtb
from spatialmath import SE3
from ir_support.robots.UTSMeshRobot import UTSMeshRobot

class DoosanM0609(UTSMeshRobot):
    """Doosan Robotics M0609 6-DOF Collaborative Robot Arm."""

    def __init__(self, base=None, mesh_dir=None):
        # Define the standard DH parameters [d, a, alpha, offset] and joint limits [qlim]
        links = [
            rtb.RevoluteDH(d=0.135, a=0.0,   alpha=pi / 2,  qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.0,   a=0.411, alpha=0.0,     offset=pi / 2, qlim=[-pi, pi]),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi / 2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.368, a=0.0,   alpha=-pi / 2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.0,   a=0.0,   alpha=pi / 2,  offset=pi,     qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0.121, a=0.0,   alpha=0.0,     qlim=[-2 * pi, 2 * pi]),
        ]

        if mesh_dir is None:
            script_dir = os.path.abspath(os.path.dirname(__file__))
            repo_root = os.path.abspath(os.path.join(script_dir, ".."))
            candidate_dirs = [
                os.path.join(repo_root, "DoosanM0609Links"),
                os.path.join(script_dir, "DoosanM0609Links"),
                os.path.join(repo_root, "DoosanM0609Links", "DoosanM0609"),
                os.path.join(script_dir, "DoosanM0609"),
                os.path.join(repo_root, "DoosanM0609"),
                script_dir,
            ]
            
            # Check explicit candidates first
            for d in candidate_dirs:
                if os.path.exists(os.path.join(d, "DoosanM0609Link0.dae")):
                    mesh_dir = os.path.abspath(d)
                    break

            # Fallback: search repo root recursively if moved anywhere else
            if mesh_dir is None:
                for root, _, files in os.walk(repo_root):
                    if "DoosanM0609Link0.dae" in files:
                        mesh_dir = root
                        break

            # Last resort
            if mesh_dir is None:
                mesh_dir = script_dir

        super().__init__(
            links=links,
            mesh_stem="DoosanM0609",
            mesh_dir=mesh_dir,
            name="DoosanM0609",
            home_q=[0.0, 0.0, pi / 2, 0.0, pi / 2, 0.0],
            base=base,
        )


if __name__ == "__main__":
    import swift
    from spatialmath import SE3

    env = swift.Swift()
    env.launch(realtime=True)

    # Instantiate and add robot to Swift environment
    robot = DoosanM0609(base=SE3(0, 0, 0))
    robot.add_to_env(env)

    # Verify motion
    robot.test()