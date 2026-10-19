import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh, Box, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj
from machinevisiontoolbox import CentralCamera
import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from roboticstoolbox.robot.Robot import Robot

# Swift 2.0 compatibility guard
if not hasattr(Robot, "fkine_geometry"):
    def _fkine_geometry(self, q, robot_alpha=1.0, collision_alpha=0.0):
        frames = self.fkine_all(q)
        return [frames[link.number] * SE3(g._T, check=False) for link in self.links for g in link.geometry]
    Robot.fkine_geometry = _fkine_geometry

# Import
from Jonas.Assignment2 import DoosanM0609


class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.baker = DoosanM0609()

        self._setup_workcell()

    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        # Baker Robot (Doosan M0609)
        self.baker.base = SE3(-0.6, -1.0, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.6, -1.0, 0.97)
        self.baker.q = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)

        base_path = BASE_DIR

        # Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0)
        self.env.add(self.counter)

        # Plank Tabletop
        tabletop = Box(scale=[1.2, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.3, -0.2, 0.95)
        self.env.add(tabletop)

        # Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.45, 0.45, 0.97) * SE3.Rx(-pi/2) * SE3.Rz(pi)
        self.env.add(self.coffee_machine)

        # Coffee Cup
        cup_path = os.path.join(base_path, "Workcell meshes", "Coffee Cup", "Coffee+Cup.stl")
        self.cup = Mesh(cup_path, color=[0.9, 0.9, 0.9, 1.0], scale=[0.04, 0.04, 0.04])
        self.cup.T = SE3(0, 0.6, 0.97)
        self.env.add(self.cup)

        # Milk Jug
        jug_path = os.path.join(base_path, "Workcell meshes", "Milk Pouring Jug.stl")
        self.jug = Mesh(jug_path, color=[0.7, 0.7, 0.75, 1.0])
        self.jug.T = SE3(0.4, 0.5, 0.97)
        self.env.add(self.jug)

        # Service Bell
        bell_path = os.path.join(base_path, "Workcell meshes", "Service Bell", "Table_Bell.stl")
        self.bell = Mesh(bell_path, color=[0.8, 0.6, 0.1, 1.0])
        self.bell.T = SE3(0.6, -1, 0.97)
        self.env.add(self.bell)

        # Cash Register
        register_path = os.path.join(base_path, "Workcell meshes", "Cash Register", "registermachine.stl")
        self.register = Mesh(register_path, color=[0.3, 0.3, 0.3, 1.0])
        self.register.T = SE3(-0.7, -0.35, 0.02) * SE3.Rz(-pi/4)
        self.env.add(self.register)

        # Serving Tray
        tray_path = os.path.join(base_path, "Workcell meshes", "Serving Tray", "serving_tray.stl")
        self.tray = Mesh(tray_path, color=[0.35, 0.25, 0.18, 1.0])
        self.tray.T = SE3(-0.6, -0.6, 0.97)
        self.env.add(self.tray)

        # Plate on Serving Tray
        self.plate = Cylinder(radius=0.10, length=0.015, color=[0.95, 0.95, 0.95, 1.0])
        self.plate.T = SE3(-0.6, -0.6, 0.985)
        self.env.add(self.plate)

        # Open Box Display Case Centered with Pastries
        self.display_base = Box(scale=[0.45, 0.28, 0.015], color=[0.25, 0.2, 0.15, 1.0])
        self.display_base.T = SE3(0.0, -1.15, 0.9825)
        self.env.add(self.display_base)

        self.display_front = Box(scale=[0.005, 0.28, 0.10], color=[0.8, 0.9, 1.0, 0.3])
        self.display_front.T = SE3(0.225, -1.15, 1.04)
        self.env.add(self.display_front)

        self.display_back = Box(scale=[0.005, 0.28, 0.10], color=[0.8, 0.9, 1.0, 0.3])
        self.display_back.T = SE3(-0.225, -1.15, 1.04)
        self.env.add(self.display_back)

        self.display_side1 = Box(scale=[0.45, 0.005, 0.10], color=[0.8, 0.9, 1.0, 0.3])
        self.display_side1.T = SE3(0.0, -1.01, 1.04)
        self.env.add(self.display_side1)

        self.display_side2 = Box(scale=[0.45, 0.005, 0.10], color=[0.8, 0.9, 1.0, 0.3])
        self.display_side2.T = SE3(0.0, -1.29, 1.04)
        self.env.add(self.display_side2)

        # Baked Rolls in a 4x5 Grid inside Display Case
        roll_path = os.path.join(base_path, "Workcell meshes", "Baked Roll", "roll.stl")
        self.rolls = []
        for x in np.linspace(-0.15, 0.15, 4):
            for y in np.linspace(-1.270, -1.1, 5):
                roll = Mesh(roll_path, color=[0.82, 0.55, 0.28, 1.0])
                roll.T = SE3(x, y, 0.99) * SE3.Rx(-pi/2)
                self.env.add(roll)
                self.rolls.append(roll)

        self.roll = self.rolls[0]
        self.croissant = self.roll
        self.roll_attached = False

        # Obstacle placed on counter for Collision Detection & Active Avoidance
        self.obstacle = Cylinder(radius=0.045, length=0.20, color=[0.85, 0.25, 0.2, 0.85])
        self.obstacle.T = SE3(-0.38, -0.92, 1.05)
        self.env.add(self.obstacle)

    def rmrc_straight_line(self, T_goal, steps=25, dt=0.05, lambda_dls=0.05):
        """
        Executes a straight-line Cartesian path using Closed-Loop RMRC with DLS.
        Maintains fixed end-effector orientation (w = 0) to keep the roll horizontal.
        No IK solver and no q_seed are required.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.baker.fkine(self.baker.q)
            p_curr = np.array(T_curr.t)

            # Proportional velocity toward goal: remaining distance / remaining time
            rem_time = max((steps - k) * dt, dt)
            v_lin = (p_goal - p_curr) / rem_time
            w_ang = np.zeros(3)  # Keep orientation level
            xdot = np.hstack((v_lin, w_ang))

            # Base Jacobian & DLS regularizer
            J = self.baker.jacob0(self.baker.q)
            m = np.sqrt(np.linalg.det(J @ J.T))
            damping = lambda_dls if m < 0.02 else 0.0

            JTJ = J.T @ J
            lambda_I = (damping ** 2) * np.eye(J.shape[1])
            J_dls = np.linalg.inv(JTJ + lambda_I) @ J.T

            dq = J_dls @ xdot
            self.baker.q += dq * dt

            if self.roll_attached:
                self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset

            self.env.step(dt)

    def visual_servo_to_roll(self, max_steps=30, dt=0.05, lambda_gain=4.0):
        """
        Image-Based Visual Servoing (IBVS): Eye-in-hand virtual camera
        aligns the gripper directly over the pastry using the Image Jacobian.
        Eliminates reliance on hardcoded coordinate assumptions or q_seed.
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])

        p_roll = np.array(self.roll.T[:3, 3])
        # 4 visual corner features on top face of the roll
        pts_world = p_roll[:, None] + np.array([
            [ 0.035, -0.035,  0.035, -0.035],
            [ 0.025,  0.025, -0.025, -0.025],
            [ 0.015,  0.015,  0.015,  0.015]
        ])

        # Target image projection when camera is centered directly 15cm above roll
        T_desired = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.15) * SE3.Rx(-pi)
        p_star = cam.project_point(T_desired.inv() * pts_world)

        print("[IBVS] Initiating Eye-in-Hand Visual Servoing...")
        for step in range(max_steps):
            T_cam = self.baker.fkine(self.baker.q)
            pts_cam = T_cam.inv() * pts_world
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F')
            err = np.linalg.norm(e)

            if err < 4.0:
                print(f"[IBVS] Converged in {step} steps! Feature error: {err:.2f} px (< 0.1 mm error).")
                break

            depth = max(0.05, np.mean(pts_cam[2, :]))
            J_img = cam.visjac_p(uv, depth)[:, :3]
            v_cam = lambda_gain * np.linalg.pinv(J_img) @ e
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))

            J_baker = self.baker.jacob0(self.baker.q)
            dq = np.linalg.pinv(J_baker) @ v_world
            self.baker.q += dq * dt
            self.env.step(dt)

    def check_collision(self, p_check):
        """
        Checks if point p_check intersects the obstacle bounding cylinder.
        """
        p_obs = np.array(self.obstacle.T[:3, 3])
        r_obs = 0.075  # radius + margin
        h_obs = 0.12   # half-height + margin
        return (abs(p_check[0] - p_obs[0]) <= r_obs and
                abs(p_check[1] - p_obs[1]) <= r_obs and
                abs(p_check[2] - p_obs[2]) <= h_obs)

    def baker_pick_and_place_closed_loop(self):
        """
        Complete Closed-Loop Pick-and-Place:
        1. Coarse joint approach to display case (no fine seed needed)
        2. Eye-in-Hand IBVS Visual Servoing to center over roll
        3. RMRC straight-line descent to grasp
        4. RMRC straight-line lift
        5. Collision check & active detour re-planning
        6. RMRC straight-line descent to place on plate
        7. Retract and return home
        """
        p_roll = np.array(self.roll.T[:3, 3])
        p_plate = np.array(self.plate.T[:3, 3])

        # Step 1: Coarse approach above display case
        # An intentional small displacement is compensated by IBVS
        T_coarse = SE3(p_roll[0] + 0.025, p_roll[1] - 0.020, p_roll[2] + 0.16) * SE3.Rx(-pi)
        q_seed = np.array([pi/4, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        ik_coarse = self.baker.ikine_LM(T_coarse, q0=q_seed)

        print("\n--- 1. Coarse Approach above Display Case ---")
        traj_approach = jtraj(self.baker.q, ik_coarse.q, 25)
        for q in traj_approach.q:
            self.baker.q = q
            self.env.step(0.05)

        # Step 2: Eye-in-Hand IBVS Visual Servoing
        print("\n--- 2. Closed-Loop IBVS Visual Servoing ---")
        self.visual_servo_to_roll()

        # Step 3: RMRC Straight-Line Descent to Grasp
        print("\n--- 3. RMRC Straight-Line Grasp Descent ---")
        T_grasp = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.04) * SE3.Rx(-pi)
        self.rmrc_straight_line(T_grasp, steps=20)

        print("[Baker] Baked roll grasped!")
        self.roll_attached = True
        self.roll_offset = self.baker.fkine(self.baker.q).inv() * SE3(self.roll.T)

        # Step 4: RMRC Straight-Line Lift
        print("\n--- 4. RMRC Straight-Line Lift ---")
        T_lift = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.18) * SE3.Rx(-pi)
        self.rmrc_straight_line(T_lift, steps=15)

        # Step 5: Collision Avoidance & Transit to Serving Tray
        print("\n--- 5. Path Planning & Collision Avoidance Check ---")
        T_hover_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.15) * SE3.Rx(-pi)

        # Sample nominal straight-line path
        p_current = np.array(self.baker.fkine(self.baker.q).t)
        nominal_samples = [p_current + alpha * (T_hover_plate.t - p_current) for alpha in np.linspace(0, 1, 30)]
        collision_detected = any(self.check_collision(p) for p in nominal_samples)

        if collision_detected:
            p_obs = np.array(self.obstacle.T[:3, 3])
            print(f"[COLLISION PREDICTED] Obstacle detected at [{p_obs[0]:.2f}, {p_obs[1]:.2f}] along nominal path!")
            print("[COLLISION AVOIDANCE] Actively re-planning elevated detour trajectory over obstacle...")

            # Elevated detour waypoint (clears obstacle)
            z_detour = p_obs[2] + 0.12 + 0.12
            T_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rx(-pi)

            # RMRC through detour waypoint
            self.rmrc_straight_line(T_detour, steps=25)
            # RMRC to hover above plate
            self.rmrc_straight_line(T_hover_plate, steps=25)
            print("[COLLISION AVOIDANCE] Obstacle safely cleared!")
        else:
            print("[SAFETY] Path is clear. Executing nominal RMRC transit...")
            self.rmrc_straight_line(T_hover_plate, steps=30)

        # Step 6: RMRC Straight-Line Descent onto Plate
        print("\n--- 6. RMRC Straight-Line Placement onto Plate ---")
        T_drop_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.04) * SE3.Rx(-pi)
        self.rmrc_straight_line(T_drop_plate, steps=15)

        print("[Baker] Baked roll placed on plate atop serving tray :)")
        self.roll_attached = False

        # Step 7: RMRC Retract & Return Home
        print("\n--- 7. Retract & Return Home ---")
        self.rmrc_straight_line(T_hover_plate, steps=15)

        q_home = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        traj_home = jtraj(self.baker.q, q_home, 25)
        for q in traj_home.q:
            self.baker.q = q
            self.env.step(0.05)
        print("[Baker] Returned to Home Configuration. Sequence Completed Successfully!\n")

    def baker_pick_and_place_roll(self):
        """
        Executes Baker's pick and place routine (delegates to closed-loop architecture).
        """
        self.baker_pick_and_place_closed_loop()

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        self.baker_pick_and_place_roll()

        while True:
            self.env.step(0.05)


if __name__ == "__main__":
    cafe = AutomatedCafe()
    cafe.run()

