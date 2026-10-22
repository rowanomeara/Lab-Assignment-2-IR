import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh, Box, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj
from machinevisiontoolbox import CentralCamera
from ir_support.robots import DobotMagician
from ir_support_extra_parts.parts import part_mesh 
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

# Import robot models
from Rowan.abb_irb120 import ABB_IRB120
from Jonas.Assignment2 import DoosanM0609

class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        
        self.barista = ABB_IRB120()
        self.baker = DoosanM0609()
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, props, and safety obstacles into the Swift environment.
        """
        base_path = BASE_DIR
        
        # Rowan's Barista Robot (ABB IRB 120)
        self.barista.base = SE3(-0.25, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.25, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)
        
        # Jonas's Baker Robot (Doosan M0609)
        self.baker.base = SE3(-0.6, -1.0, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.6, -1.0, 0.97)
        self.baker.q = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)
        
        # Dobot Magician (Milk Pouring)
        dobot_base = SE3(-0.9, 0.0, 0.97)
        self.dobot = DobotMagician(base=dobot_base)
        self.dobot.base = dobot_base
        
        # Make the milk jug level:
        q2_ready = 25 * pi / 180
        q3_ready = 20 * pi / 180
        q4_level = -(q2_ready + q3_ready)
        self.dobot_q_ready = np.array([0.0, q2_ready, q3_ready, q4_level, 0.0])
        self.dobot.q = self.dobot_q_ready
        
        # Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)
        
        # Plank Tabletop
        tabletop = Box(scale=[1.8, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.6, -0.2, 0.95)
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
        self.jug = Mesh(jug_path, color=[0.7, 0.7, 0.75, 1.0], scale=[0.001, 0.001, 0.001])
        self.jug_offset = SE3.Ry(pi) * SE3(-0.168, -0.091, -0.081)
        self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
        self.env.add(self.jug)
        self.dobot.add_to_env(self.env)
        
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
        self.tray = Mesh(tray_path, color=[0.0, 0.0, 0.0, 1.0])
        self.tray.T = SE3(-0.3, -0.3, 0.97)
        self.env.add(self.tray)

        # Plate on Serving Tray
        self.plate = Cylinder(radius=0.10, length=0.015, color=[0.95, 0.95, 0.95, 1.0])
        self.plate.T = SE3(-0.38, -0.3, 0.985)
        self.env.add(self.plate)

        # Open Box Display Case     
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
        
        # Environment Props - Cafe Floor & Walls
        floor = Box(scale=[6.0, 7.0, 0.02], color=[0.25, 0.18, 0.12, 1.0])
        floor.T = SE3(0.0, -0.5, 0.001)
        self.env.add(floor)
        
        back_wall = Box(scale=[0.05, 5.0, 2.6], color=[0.85, 0.82, 0.78, 1.0])
        back_wall.T = SE3(-1.8, -0.5, 1.3)
        self.env.add(back_wall)
        
        menu_board = Box(scale=[0.02, 1.4, 0.8], color=[0.12, 0.12, 0.12, 1.0])
        menu_board.T = SE3(-1.75, 0.2, 1.9)
        self.env.add(menu_board)
        
        table = part_mesh("StandingBarTable")
        table.T = SE3(2.1, 2.5, 0.0)
        self.env.add(table)

        chair = part_mesh("chair")
        chair.T = SE3(2.5, 2.5, 0.0) * SE3.Ry(-pi / 2)* SE3.Rz(-pi / 2)
        self.env.add(chair)
        
        customer = part_mesh("personMaleCasual") 
        customer.T = SE3(1.5, 0.5, 0.0) * SE3.Rz(pi)
        self.env.add(customer)
        
        takeaway_cup = part_mesh("TakeawayCup")
        takeaway_cup.T = SE3(0.5, -0.1, 0.97)
        self.env.add(takeaway_cup)
        
        # --- DELIBERATE SAFETY OBSTACLES FOR ACTIVE COLLISION AVOIDANCE ---
        # 1. Barista Obstacle: Cleaning spray bottle in front of Barista
        self.bottle = Cylinder(radius=0.045, length=0.15, color=[0.9, 0.2, 0.2, 0.85])
        self.bottle.T = SE3(-0.15, -0.05, 0.97 + 0.1)
        self.env.add(self.bottle)

        # 2. Baker Obstacle: Syrup dispenser bottle placed on counter along transit path
        self.baker_obstacle = Cylinder(radius=0.045, length=0.20, color=[0.85, 0.25, 0.2, 0.85])
        self.baker_obstacle.T = SE3(-0.28, -0.78, 1.05)
        self.env.add(self.baker_obstacle)

    # =========================================================================
    # BARISTA ADVANCED CONTROL (RMRC, IBVS, COLLISION AVOIDANCE)
    # =========================================================================

    def barista_rmrc_straight_line(self, T_goal, steps=25, dt=0.05):
        """
        Closed-Loop RMRC with DLS for the ABB IRB 120 Barista robot.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.barista.fkine(self.barista.q)
            rem_time = max((steps - k) * dt, dt)
            v_lin = (p_goal - np.array(T_curr.t)) / rem_time
            xdot = np.hstack((v_lin, np.zeros(3)))

            J = self.barista.jacob0(self.barista.q)
            m = np.sqrt(np.linalg.det(J @ J.T))
            damping = 0.001 if m < 0.01 else 0.0
            dq = np.linalg.inv(J.T @ J + damping * np.eye(6)) @ J.T @ xdot

            self.barista.q += dq * dt

            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                if hasattr(self, 'coffee'):
                    self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)

            self.env.step(dt)

    def visual_servo_to_cup(self, max_steps=30, dt=0.05):
        """
        Eye-in-Hand IBVS: Centers Barista gripper directly over the cup using Image Jacobian.
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])

        p = SE3(self.cup.T).t
        pts_world = p[:, None] + [[0.04, -0.04, 0.04, -0.04], [0, 0, 0, 0], [0.08, 0.08, 0.01, 0.01]]

        T_desired = SE3(self.cup.T) * SE3(0.0, -0.15, 0.05) * SE3.Rx(-pi/2)
        p_star = cam.project_point(T_desired.inv() * pts_world)

        print("\n--- [Barista] 2. Closed-Loop IBVS Visual Servoing ---")
        for step in range(max_steps):
            T_cam = self.barista.fkine(self.barista.q)
            pts_cam = T_cam.inv() * pts_world
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F')

            if np.linalg.norm(e) < 5.0:
                print(f"[Barista IBVS] Visual servoing converged in {step} steps! Feature error: {np.linalg.norm(e):.2f} px")
                break

            depth = max(0.05, np.mean(pts_cam[2, :]))
            J_img = cam.visjac_p(uv, depth)[:, :3]
            v_cam = 3.5 * np.linalg.pinv(J_img) @ e
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))
            dq = np.linalg.pinv(self.barista.jacob0(self.barista.q)) @ v_world

            self.barista.q += dq * dt
            self.env.step(dt)

    def barista_pick_up_cup(self):
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.15, 0.16) * SE3.Rx(-pi/2)
        ik_hover = self.barista.ikine_LM(hover_pose, q0=self.barista.q)
        traj_hover = jtraj(self.barista.q, ik_hover.q, 20)
        
        print("\n--- [Barista] 1. Coarse Approach to Cup ---")
        for q_step in traj_hover.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
        # 1. Active Eye-in-Hand IBVS Visual Servoing
        self.visual_servo_to_cup()
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        ik_raise  = self.barista.ikine_LM(hover_pose, q0=ik_target.q)   
        
        traj_target = jtraj(self.barista.q, ik_target.q, 15)
        traj_raise  = jtraj(ik_target.q,    ik_raise.q, 15)
            
        # Lower to Cup
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
        print("[Barista] Cup grasped!")
        self.cup_attached = True
        self.cup_offset = self.barista.fkine(self.barista.q).inv() * SE3(self.cup.T)
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
            
    def barista_place_cup(self):
        target_pose = SE3(0.27, 0.22, 1.13) * SE3.Rx(-pi/2)
        
        print("\n--- [Barista] 3. RMRC Straight-Line Positioning under Coffee Machine ---")
        self.barista_rmrc_straight_line(target_pose, steps=25)
        
        print("[Barista] Cup positioned under coffee machine :)")
        self.cup_attached = False
        
        q_retract = self.barista.q + [0, -50 * pi / 180, 80 * pi / 180, 0, 0, 0]
        traj_retract = jtraj(self.barista.q, q_retract, 20)
            
        for q_step in traj_retract.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        coffee_color = [0.15, 0.08, 0.03, 1.0]
        if not hasattr(self, 'coffee'):
            self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.add(self.coffee)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q, ik_target.q, 20)
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
        
        traj_raise = jtraj(self.barista.q, self.barista.q + [0, -50 * pi / 180, 80 * pi / 180, 0, 0, -25 * pi / 180], 20)
        print("[Barista] Cup retrieved with fresh coffee!")
        self.cup_attached = True
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
        traj_rot = jtraj(self.barista.q, self.barista.q + [-100 * pi / 180, 0, 0, 0, 0, 0], 30)
        for q_step in traj_rot.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
    def barista_present_cup_to_dobot(self):
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        ee_target = handover_cup_pose * self.cup_offset.inv()
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q)
        
        traj_present = jtraj(self.barista.q, ik_present.q, 25)
        for q_step in traj_present.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
                
        print("[Barista] Cup presented to Dobot for milk :)")

    def dobot_pour_milk(self):
        def move(q_target, steps):
            for q in jtraj(self.dobot.q, q_target, steps).q:
                self.dobot.q = q
                self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
                self.env.step(0.05)

        T_cup_rim = (SE3(self.cup.T) * SE3(-0.05, 0.0, 0.125)).t
        T_ee_target = SE3(T_cup_rim - [0.156, 0.0, 0.080]) * SE3.Rx(pi)

        sol_reach = self.dobot.ikine_LM(T_ee_target, q0=self.dobot.q, mask=[1, 1, 1, 0, 0, 0])
        q_reach = sol_reach.q.copy()
        q_reach[3] = -(q_reach[1] + q_reach[2])

        move(q_reach, 20)

        q_pour = q_reach.copy()
        q_pour[3] += 25 * pi / 180
        move(q_pour, 15)

        if not hasattr(self, 'milk'):
            self.milk = Cylinder(radius=0.058, length=0.015, color=[0.55, 0.36, 0.20, 1.0])
            self.env.add(self.milk)
        self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
        print("[Dobot] Poured milk into the cup <3")

        move(q_reach, 15)
        move(self.dobot_q_ready, 20)
        
    def barista_serve_coffee(self):
        target_cup = SE3(self.tray.T) * SE3(0.08, 0.0, 0.03)
        hover_cup = SE3(0, 0, 0.12) * target_cup

        ee_hover = hover_cup * SE3.Rz(pi) * self.cup_offset.inv()
        ee_target = target_cup * SE3.Rz(pi) * self.cup_offset.inv()

        ik_hover = self.barista.ikine_LM(ee_hover, q0=self.barista.q)
        ik_serve = self.barista.ikine_LM(ee_target, q0=ik_hover.q)
        ik_retract = self.barista.ikine_LM(ee_hover, q0=ik_serve.q)

        # Barista Active Collision Detection & Detour Re-planning
        p_obs = SE3(self.bottle.T).t
        r_obs = getattr(self.bottle, 'radius', 0.045) + 0.06
        h_obs = getattr(self.bottle, 'length', 0.15) / 2 + 0.06

        in_box = lambda p: (p_obs[0] - r_obs <= p[0] <= p_obs[0] + r_obs) and \
                           (p_obs[1] - r_obs <= p[1] <= p_obs[1] + r_obs) and \
                           (p_obs[2] - h_obs <= p[2] <= p_obs[2] + h_obs)

        traj_nominal = jtraj(self.barista.q, ik_hover.q, 30)
        collision = any(in_box((self.barista.fkine(q) * self.cup_offset).t) for q in traj_nominal.q)

        print("\n--- [Barista] 4. Service Transit & Collision Avoidance Check ---")
        if collision:
            print(f"[Barista Safety] Collision predicted with obstacle at {np.round(p_obs[:2], 2)}! Actively re-planning detour...")
            z_detour = p_obs[2] + getattr(self.bottle, 'length', 0.15) / 2 + 0.23
            ee_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rz(pi/2) * self.cup_offset.inv()
            q_detour = self.barista.ikine_LM(ee_detour, q0=self.barista.q).q

            traj_to_detour = jtraj(self.barista.q, q_detour, 20).q
            traj_to_hover  = jtraj(q_detour, ik_hover.q, 20).q
            traj_hover_q = np.vstack((traj_to_detour, traj_to_hover))
            print("[Barista Safety] Executing elevated detour over obstacle...")
        else:
            print("[Barista Safety] Path clear. Proceeding along nominal trajectory...")
            traj_hover_q = traj_nominal.q

        traj_serve = jtraj(ik_hover.q, ik_serve.q, 15)
        traj_retract = jtraj(ik_serve.q, ik_retract.q, 15)

        for q_step in traj_hover_q:
            self.barista.q = q_step
            self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        for q_step in traj_serve.q:
            self.barista.q = q_step
            self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        print("[Barista] Coffee placed on tray :)")
        self.cup_attached = False

        for q_step in traj_retract.q:
            self.barista.q = q_step
            self.env.step(0.05)

        q_home = np.array([-pi/3, 0.0, 20 * pi / 180, 0.0, 30 * pi / 180, 0.0])
        traj_home = jtraj(self.barista.q, q_home, 20)
        for q_step in traj_home.q:
            self.barista.q = q_step
            self.env.step(0.05)

    # =========================================================================
    # BAKER ADVANCED CONTROL (RMRC, IBVS, COLLISION AVOIDANCE)
    # =========================================================================

    def baker_rmrc_straight_line(self, T_goal, steps=25, dt=0.05, lambda_dls=0.05):
        """
        Closed-Loop RMRC with DLS for the Doosan M0609 Baker robot.
        Maintains fixed end-effector orientation (w = 0) to keep the roll horizontal.
        No IK solver and no q_seed are required.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.baker.fkine(self.baker.q)
            p_curr = np.array(T_curr.t)

            rem_time = max((steps - k) * dt, dt)
            v_lin = (p_goal - p_curr) / rem_time
            w_ang = np.zeros(3)  # Keep orientation level
            xdot = np.hstack((v_lin, w_ang))

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
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])

        p_roll = np.array(self.roll.T[:3, 3])
        pts_world = p_roll[:, None] + np.array([
            [ 0.035, -0.035,  0.035, -0.035],
            [ 0.025,  0.025, -0.025, -0.025],
            [ 0.015,  0.015,  0.015,  0.015]
        ])

        T_desired = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.15) * SE3.Rx(-pi)
        p_star = cam.project_point(T_desired.inv() * pts_world)

        print("[Baker IBVS] Initiating Eye-in-Hand Visual Servoing...")
        for step in range(max_steps):
            T_cam = self.baker.fkine(self.baker.q)
            pts_cam = T_cam.inv() * pts_world
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F')
            err = np.linalg.norm(e)

            if err < 4.0:
                print(f"[Baker IBVS] Converged in {step} steps! Feature error: {err:.2f} px (< 0.1 mm error).")
                break

            depth = max(0.05, np.mean(pts_cam[2, :]))
            J_img = cam.visjac_p(uv, depth)[:, :3]
            v_cam = lambda_gain * np.linalg.pinv(J_img) @ e
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))

            J_baker = self.baker.jacob0(self.baker.q)
            dq = np.linalg.pinv(J_baker) @ v_world
            self.baker.q += dq * dt
            self.env.step(dt)

    def check_baker_collision(self, p_check):
        """
        Checks if point p_check intersects the Baker obstacle bounding cylinder.
        """
        p_obs = np.array(self.baker_obstacle.T[:3, 3])
        r_obs = 0.075
        h_obs = 0.12
        return (abs(p_check[0] - p_obs[0]) <= r_obs and
                abs(p_check[1] - p_obs[1]) <= r_obs and
                abs(p_check[2] - p_obs[2]) <= h_obs)

    def baker_pick_and_place_roll(self):
        """
        Baker robot picks up the baked roll and places it onto the plate on the serving tray
        using Closed-Loop IBVS Visual Servoing, RMRC Straight-Line motions, and Active Collision Avoidance.
        """
        p_roll = np.array(self.roll.T[:3, 3])
        p_plate = np.array(self.plate.T[:3, 3])

        # Step 1: Coarse approach above display case
        T_coarse = SE3(p_roll[0] + 0.025, p_roll[1] - 0.020, p_roll[2] + 0.16) * SE3.Rx(-pi)
        q_seed = np.array([pi/4, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        ik_coarse = self.baker.ikine_LM(T_coarse, q0=q_seed)

        print("\n--- [Baker] 1. Coarse Approach above Display Case ---")
        traj_approach = jtraj(self.baker.q, ik_coarse.q, 25)
        for q in traj_approach.q:
            self.baker.q = q
            self.env.step(0.05)

        # Step 2: Eye-in-Hand IBVS Visual Servoing
        print("--- [Baker] 2. Closed-Loop IBVS Visual Servoing ---")
        self.visual_servo_to_roll()

        # Step 3: RMRC Straight-Line Descent to Grasp
        print("--- [Baker] 3. RMRC Straight-Line Grasp Descent ---")
        T_grasp = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.04) * SE3.Rx(-pi)
        self.baker_rmrc_straight_line(T_grasp, steps=20)

        print("[Baker] Baked roll grasped!")
        self.roll_attached = True
        self.roll_offset = self.baker.fkine(self.baker.q).inv() * SE3(self.roll.T)

        # Step 4: RMRC Straight-Line Lift
        print("--- [Baker] 4. RMRC Straight-Line Lift ---")
        T_lift = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.18) * SE3.Rx(-pi)
        self.baker_rmrc_straight_line(T_lift, steps=15)

        # Step 5: Collision Avoidance & Transit to Serving Tray
        print("--- [Baker] 5. Path Planning & Collision Avoidance Check ---")
        T_hover_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.15) * SE3.Rx(-pi)

        p_current = np.array(self.baker.fkine(self.baker.q).t)
        nominal_samples = [p_current + alpha * (T_hover_plate.t - p_current) for alpha in np.linspace(0, 1, 30)]
        collision_detected = any(self.check_baker_collision(p) for p in nominal_samples)

        if collision_detected:
            p_obs = np.array(self.baker_obstacle.T[:3, 3])
            print(f"[Baker Safety] Collision predicted with obstacle at [{p_obs[0]:.2f}, {p_obs[1]:.2f}] along nominal path!")
            print("[Baker Safety] Actively re-planning elevated detour trajectory over obstacle...")

            z_detour = p_obs[2] + 0.12 + 0.12
            T_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rx(-pi)

            self.baker_rmrc_straight_line(T_detour, steps=25)
            self.baker_rmrc_straight_line(T_hover_plate, steps=25)
            print("[Baker Safety] Obstacle safely cleared!")
        else:
            print("[Baker Safety] Path clear. Executing nominal RMRC transit...")
            self.baker_rmrc_straight_line(T_hover_plate, steps=30)

        # Step 6: RMRC Straight-Line Descent onto Plate
        print("--- [Baker] 6. RMRC Straight-Line Placement onto Plate ---")
        T_drop_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.04) * SE3.Rx(-pi)
        self.baker_rmrc_straight_line(T_drop_plate, steps=15)

        print("[Baker] Baked roll placed on plate atop serving tray :)")
        self.roll_attached = False

        # Step 7: RMRC Retract & Return Home
        print("--- [Baker] 7. Retract & Return Home ---")
        self.baker_rmrc_straight_line(T_hover_plate, steps=15)

        q_home = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        traj_home = jtraj(self.baker.q, q_home, 25)
        for q in traj_home.q:
            self.baker.q = q
            self.env.step(0.05)
        print("[Baker] Returned to Home Configuration. Sequence Completed Successfully!\n")

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        
        # 1. Barista executes coffee preparation with IBVS & RMRC
        self.barista_pick_up_cup()
        self.barista_place_cup()
        self.barista_get_coffee()

        # 2. Dobot milk pouring interaction
        self.barista_present_cup_to_dobot()
        self.dobot_pour_milk()
        
        # 3. Barista delivers coffee with Active Collision Avoidance
        self.barista_serve_coffee()

        # 4. Baker executes pastry delivery with IBVS, RMRC, & Active Collision Avoidance
        self.baker_pick_and_place_roll()
        
        while True:
            self.env.step(0.05)

if __name__ == "__main__":
    cafe = AutomatedCafe()
    cafe.run()
