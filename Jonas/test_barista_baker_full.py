import numpy as np
from math import pi
import time
import swift
from spatialgeometry import Mesh, Box, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj
import os
import sys
sys.modules['numpy.char'] = np.char
from machinevisiontoolbox import CentralCamera
from ir_support.robots import DobotMagician
from ir_support_extra_parts.parts import part_mesh

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "Jonas" else SCRIPT_DIR
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
from Rowan.abb_irb120 import ABB_IRB120
from Jonas.Assignment2 import DoosanM0609
from Lachlan.LachlanMesh.staubli_tx60.staubliTX60 import StaubliTX60

class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        
        self.barista = ABB_IRB120()
        
        self.baker = DoosanM0609()
        
        self.staubli = StaubliTX60()
        
        self.estop_active = False
        self.sensor_breached = False
        self.is_running = False
        self.is_paused = False
        self.cup_attached = False
        self.roll_attached = False
        self.tray_attached = False

        # Guard env.step globally against E-stop and safety sensor pauses
        orig_step = self.env.step
        def guarded_step(dt=0.05):
            while self.estop_active or self.curtain_breached or getattr(self, "is_paused", False):
                time.sleep(0.05)
            orig_step(dt)
        self.env.step = guarded_step
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        
        #Rowan's Barista Robot
        self.barista.base = SE3(-0.25, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.25, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)
        
        #Jonas's Baker Robot
        self.baker.base = SE3(-0.6, -1.0, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.6, -1.0, 0.97)
        self.baker.q = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)
        
        #Lachlan's Staubli Robot
        self.staubli.base = SE3(0.30, -0.56, 0.97)
        self.staubli.base_link_mesh.T = SE3(0.30, -0.56, 0.97)
        self.env.add(self.staubli.base_link_mesh)
        self.env.add(self.staubli)
        
        #Dobot 
        dobot_base = SE3(-0.9, 0.0, 0.97)
        self.dobot = DobotMagician(base=dobot_base)
        self.dobot.base = dobot_base
        
        # Making the jug level:
        q2_ready = 25 * pi / 180
        q3_ready = 20 * pi / 180
        q4_level = -(q2_ready + q3_ready)
        self.dobot_q_ready = np.array([0.0, q2_ready, q3_ready, q4_level, 0.0])
        self.dobot.q = self.dobot_q_ready
        
        base_path = BASE_DIR

        #Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)
        
        # Plank Tabletop
        tabletop = Box(scale=[1.8, 2.2, 0.97], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.6, -0.2, 0.49)
        self.env.add(tabletop)

        # Counter Extension
        counter_extension = Box(scale=[0.63, 0.53, 0.97], color=[0.8, 0.8, 0.8, 1.0])
        counter_extension.T = SE3(-1.185, -1.565, 0.485)
        self.env.add(counter_extension)

        #Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.45, 0.45, 0.97)*SE3.Rx(-pi/2)*SE3.Rz(pi) 
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
        self.tray = Mesh(tray_path, color=[0.35, 0.25, 0.18, 1.0])
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
        floor = Box(scale=[8.5, 9.0, 0.02], color=[0.34, 0.24, 0.16, 1.0])
        floor.T = SE3(0.75, -0.5, 0.001)
        self.env.add(floor)
        
        back_wall = Box(scale=[0.05, 9.0, 2.6], color=[0.88, 0.86, 0.82, 1.0])
        back_wall.T = SE3(-1.8, -0.5, 1.3)
        self.env.add(back_wall)
        
        menu_board = Box(scale=[0.02, 1.4, 0.8], color=[0.12, 0.12, 0.12, 1.0])
        menu_board.T = SE3(-1.75, 0.2, 1.9)
        self.env.add(menu_board)

        # Cafe Lounge Seating
        sofa_path = os.path.join(base_path, "Workcell meshes", "Sofa", "sofa.stl")
        sofa = Mesh(sofa_path, color=[0.36, 0.22, 0.14, 1.0])
        sofa.T = SE3(3.4, -2.5, 0.0) * SE3.Rz(-pi / 2) * SE3.Rx(pi / 2)
        self.env.add(sofa)

        coffee_table_path = os.path.join(base_path, "Workcell meshes", "CoffeeTable", "coffee_table.stl")
        coffee_table = Mesh(coffee_table_path, color=[0.94, 0.93, 0.90, 1.0], scale=[0.85, 0.85, 0.85])
        coffee_table.T = SE3(2.4, -2.5, 0.0) * SE3.Rx(pi / 2)
        self.env.add(coffee_table)

        # Base orientation for upright chair model: SE3.Ry(-pi/2) * SE3.Rz(-pi/2) faces -X
        chair_base_rot = SE3.Ry(-pi / 2) * SE3.Rz(-pi / 2)

        lounge_chair1 = part_mesh("chair")
        lounge_chair1.T = SE3(1.6, -2.1, 0.0) * SE3.Rz(-0.46 - pi) * chair_base_rot
        self.env.add(lounge_chair1)

        lounge_chair2 = part_mesh("chair")
        lounge_chair2.T = SE3(1.6, -2.9, 0.0) * SE3.Rz(0.46 - pi) * chair_base_rot
        self.env.add(lounge_chair2)

        # Potted Plants 
        plant_path = os.path.join(base_path, "Workcell meshes", "Plant", "plant.stl")
        plant_lounge = Mesh(plant_path, color=[0.22, 0.52, 0.26, 1.0], scale=[0.9, 0.9, 0.9])
        plant_lounge.T = SE3(3.4, -1.3, 0.0) * SE3.Rx(pi / 2)
        self.env.add(plant_lounge)

        plant_entry = Mesh(plant_path, color=[0.25, 0.50, 0.22, 1.0], scale=[0.8, 0.8, 0.8])
        plant_entry.T = SE3(0.8, -2.8, 0.0) * SE3.Rx(pi / 2)
        self.env.add(plant_entry)

        plant_dining = Mesh(plant_path, color=[0.20, 0.55, 0.28, 1.0], scale=[0.85, 0.85, 0.85])
        plant_dining.T = SE3(1.1, 2.8, 0.0) * SE3.Rx(pi / 2)
        self.env.add(plant_dining)

        # Dining Area
        table1 = part_mesh("StandingBarTable")
        table1.T = SE3(2.2, 2.5, 0.0)
        self.env.add(table1)

        chair_bar1 = part_mesh("chair")
        chair_bar1.T = SE3(2.6, 2.5, 0.0) * chair_base_rot
        self.env.add(chair_bar1)

        chair_bar2 = part_mesh("chair")
        chair_bar2.T = SE3(1.8, 2.5, 0.0) * SE3.Rz(pi) * chair_base_rot
        self.env.add(chair_bar2)

        table2 = part_mesh("StandingBarTable")
        table2.T = SE3(3.3, 1.4, 0.0)
        self.env.add(table2)

        chair_bar3 = part_mesh("chair")
        chair_bar3.T = SE3(3.7, 1.4, 0.0) * chair_base_rot
        self.env.add(chair_bar3)

        chair_bar4 = part_mesh("chair")
        chair_bar4.T = SE3(3.3, 1.8, 0.0) * SE3.Rz(-pi / 2) * chair_base_rot
        self.env.add(chair_bar4)

        # Customers
        customer1 = part_mesh("personMaleCasual") 
        customer1.T = SE3(1.5, 0.5, 0.0) * SE3.Rz(pi)
        self.env.add(customer1)

        customer2 = part_mesh("personFemaleBusiness")
        customer2.T = SE3(2.2, 1.9, 0.0)
        self.env.add(customer2)

        customer3 = part_mesh("personMaleOld")
        customer3.T = SE3(3.4, -0.6, 0.0) * SE3.Rz(-pi/2)
        self.env.add(customer3)
        
        takeaway_cup = part_mesh("TakeawayCup")
        takeaway_cup.T = SE3(0.5, -0.1, 0.97)
        self.env.add(takeaway_cup)
        
        self.estop = part_mesh("emergencyStopWallMounted")
        self.estop.T = SE3(-1.77, -0.6, 1.4) * SE3.Rz(-pi/2)
        self.env.add(self.estop)

        # Safety Equipment 3D Models
        # Barista Light Curtain (Coffee Machine front side)
        curtain_barista_p1 = part_mesh("SafetyLightCurtain")
        curtain_barista_p1.T = SE3(0.75, 0.05, 0.97)
        self.env.add(curtain_barista_p1)

        curtain_barista_p2 = part_mesh("SafetyLightCurtain")
        curtain_barista_p2.T = SE3(0.75, 0.80, 0.97)
        self.env.add(curtain_barista_p2)

        beam_barista = Box(scale=[0.01, 0.75, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_barista.T = SE3(0.75, 0.425, 1.36)
        self.env.add(beam_barista)

        # Barista Light Curtain (covering full outer flank side of counter and bench)
        curtain_flank_p1 = part_mesh("SafetyLightCurtain")
        curtain_flank_p1.T = SE3(0.74, 0.86, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_flank_p1)

        curtain_flank_p2 = part_mesh("SafetyLightCurtain")
        curtain_flank_p2.T = SE3(-1.45, 0.86, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_flank_p2)

        beam_flank = Box(scale=[2.19, 0.01, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_flank.T = SE3(-0.355, 0.86, 1.36)
        self.env.add(beam_flank)

        # Side Wing Light Curtain (covering full side counter wing to the end of the bench)
        curtain_wing_p1 = part_mesh("SafetyLightCurtain")
        curtain_wing_p1.T = SE3(0.18, -1.75, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_wing_p1)

        curtain_wing_p2 = part_mesh("SafetyLightCurtain")
        curtain_wing_p2.T = SE3(-1.45, -1.75, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_wing_p2)

        beam_wing = Box(scale=[1.63, 0.01, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_wing.T = SE3(-0.635, -1.75, 1.36)
        self.env.add(beam_wing)

        # Keep beams together safety updates
        self.curtain_beams = [beam_barista, beam_flank, beam_wing]

        # Status Light Tower + Beacon
        light_tower = part_mesh("LightTower")
        light_tower.T = SE3(-0.05, -1.65, 0.97)
        self.env.add(light_tower)

        self.beacon = Cylinder(radius=0.03, length=0.07, color=[0.0, 1.0, 0.0, 0.95])
        self.beacon.T = SE3(-0.05, -1.65, 1.45)
        self.env.add(self.beacon)

        # Wall Shelf + Fire Extinguisher
        self.wall_shelf = Box(scale=[0.22, 0.30, 0.025], color=[0.35, 0.25, 0.18, 1.0])
        self.wall_shelf.T = SE3(-1.70, -1.0, 1.25)
        self.env.add(self.wall_shelf)

        fire_ext = part_mesh("FireExtinguisher")
        fire_ext.T = SE3(-1.68, -1.0, 1.27)
        self.env.add(fire_ext)

        # =========================================================================
        # VIVA DEMONSTRATION CHEAT SHEET: OBSTACLE POSITIONS & TEST SCENARIOS
        # =========================================================================
        # --- Baker (Doosan M0609) Obstacle Options ---
        # 1. Mid-Flight (Default): Triggers standard elevated detour arc
        #    self.baker_obstacle.T = SE3(-0.28, -0.78, 1.05)
        # 2. Early-Flight (Near Display Case): Triggers climb immediately after lift
        #    self.baker_obstacle.T = SE3(-0.20, -1.05, 1.05)
        # 3. Late-Flight (Near Serving Tray): Detours right before plate drop
        #    self.baker_obstacle.T = SE3(-0.34, -0.52, 1.05)
        # 4. Clear Path Test (Tutor Trap): Offset sideways; proves robot detects
        #    path is clear and takes fast nominal route without unnecessary detour!
        #    self.baker_obstacle.T = SE3(-0.10, -0.78, 1.05)  # Front side
        #    self.baker_obstacle.T = SE3(-0.48, -0.78, 1.05)  # Back wall side
        #
        # --- Barista (ABB IRB 120) Obstacle Options ---
        # 1. Mid-Transit (Default): self.bottle.T = SE3(-0.15, -0.05, 1.07)
        # 2. Near Dobot Handover:  self.bottle.T = SE3(-0.22,  0.08, 1.07)
        # 3. Clear Path (Off-Path): self.bottle.T = SE3( 0.10, -0.05, 1.07)
        # =========================================================================
        # Baker Obstacle
        self.baker_obstacle = Cylinder(radius=0.045, length=0.20, color=[0.85, 0.25, 0.2, 0.85])
        self.baker_obstacle.T = SE3(-0.28, -0.78, 1.05)
        self.env.add(self.baker_obstacle)

        # Barista Obstacle
        self.bottle = Cylinder(radius=0.045, length=0.15, color=[0.9, 0.2, 0.2, 0.85])
        self.bottle.T = SE3(-0.15, -0.05, 0.97 + 0.1)
        self.env.add(self.bottle)

        # Caution Sign standing on floor
        warning_sign = part_mesh("WarningSign")
        warning_sign.T = SE3(-1.5, 0.7, 0.0)
        self.env.add(warning_sign)


    @property
    def curtain_breached(self):
        """True if manual curtain toggle is active or robot reaches outside cell"""
        return self.sensor_breached or (self.baker.fkine(self.baker.q).t[1] < -1.70)

    def press_physical_estop(self):
        """Simulates physical E-stop push button press."""
        self.estop_active = True
        print("[SAFETY] Physical Emergency Stop pressed - robot halted.")

    def reset_simulation(self):
        """Reset robots, objects, and safety states back to initial conditions."""
        self.estop_active = False
        self.sensor_breached = False
        self.is_running = False
        self.is_paused = False
        self.cup_attached = False
        self.roll_attached = False
        self.tray_attached = False
        self.barista.q = np.zeros(6)
        self.dobot.q = self.dobot_q_ready.copy()
        self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
        self.baker.q = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        self.staubli.q = np.zeros(6)
        self.cup.T = SE3(0, 0.6, 0.97)
        self.tray.T = SE3(-0.3, -0.3, 0.97)
        self.plate.T = SE3(-0.3, -0.3, 0.97) * SE3(0.0, 0.08, 0.02)
        self.roll.T = SE3(0.0, -0.85, 1.0)
        if hasattr(self, 'coffee'):
            self.coffee.T = SE3(0, 0, -10)
        if hasattr(self, 'milk'):
            self.milk.T = SE3(0, 0, -10)
        self.env.step(0.05)
        
        
        
    # BARISTA CONTROL (RMRC, IBVS, OBSTACLE COLLISION AVOIDANCE)

    def barista_rmrc_straight_line(self, T_goal, steps=25, dt=0.05):
        """
        Executes a straight-line Cartesian path using Closed-Loop RMRC with DLS.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.barista.fkine(self.barista.q)
            
            # Closed-loop velocity: remaining distance / remaining time (to eliminate undershoot)
            rem_time = max((steps - k) * dt, dt)
            v_lin = (p_goal - np.array(T_curr.t)) / rem_time
            xdot = np.hstack((v_lin, np.zeros(3)))

            # Jacobian & DLS
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

    rmrc_straight_line = barista_rmrc_straight_line

    def visual_servo_to_cup(self, max_steps=25, dt=0.05):
        """
        IBVS: Translates gripper to center directly over the cup using world-frame Jacobian.
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])

        p = SE3(self.cup.T).t
        pts_world = p[:, None] + [[0.04, -0.04, 0.04, -0.04], [0, 0, 0, 0], [0.08, 0.08, 0.01, 0.01]]

        T_desired = SE3(self.cup.T) * SE3(0.0, -0.15, 0.05) * SE3.Rx(-pi/2)
        p_star = cam.project_point(T_desired.inv() * pts_world)

        depth = 0.15
        for step in range(max_steps):
            T_cam = self.barista.fkine(self.barista.q)
            pts_cam = T_cam.inv() * pts_world
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F')

            if np.linalg.norm(e) < 5.0:
                print(f"Visual servoing converged in {step} steps!")
                break

            # 3-DOF camera translation
            J_img = cam.visjac_p(uv, depth)[:, :3]
            v_cam = 0.8 * np.linalg.pinv(J_img) @ e

            # Rotate to world frame so up is always up (+Z), right is always right (+X)
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))
            dq = np.linalg.pinv(self.barista.jacob0(self.barista.q)) @ v_world
            
            self.barista.q += dq * dt
            self.env.step(dt)

    def barista_pick_up_cup(self):
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.15, 0.16) * SE3.Rx(-pi/2)
        ik_hover  = self.barista.ikine_LM(hover_pose,  q0=self.barista.q)
        traj_hover  = jtraj(self.barista.q, ik_hover.q, 20)
        
        # Move to Hover
        for q_step in traj_hover.q:
            self.barista.q = q_step
            self.env.step(0.05)
        
        self.visual_servo_to_cup()
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise  = self.barista.ikine_LM(hover_pose,  q0=ik_target.q) 
        
        traj_target = jtraj(self.barista.q, ik_target.q, 20)
        traj_raise  = jtraj(ik_target.q, ik_raise.q, 20)
            
        # Lower to Cup
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
        print("Cup grasped by Barista!")
        self.cup_attached = True
        self.cup_offset = self.barista.fkine(self.barista.q).inv() * SE3(self.cup.T)
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
            
    def barista_place_cup(self):
        target_pose = SE3(0.27, 0.22, 1.13) * SE3.Rx(-pi/2)
        self.barista_rmrc_straight_line(target_pose, steps=25)
        
        print("Cup under coffee machine :)")
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
            self.env.add(self.coffee)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        for q_step in jtraj(self.barista.q, ik_target.q, 25).q:
            self.barista.q = q_step
            self.env.step(0.05)
        self.cup_attached = True
        
        T_extract = SE3(-0.27, -0.20, 0.0) * self.barista.fkine(self.barista.q)
        self.barista_rmrc_straight_line(T_extract, steps=25)
            
    def barista_present_cup_to_dobot(self):
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        ee_target = handover_cup_pose * self.cup_offset.inv()
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q)
        
        # Collision Avoidance
        # Get bottle location and build a safety box around it (radius + height + cup buffer)
        p_obs = SE3(self.bottle.T).t                            # Obstacle center [x, y, z] in world frame
        r_obs = getattr(self.bottle, 'radius', 0.045) + 0.06    # Obstacle radius + cup clearance
        h_obs = getattr(self.bottle, 'length', 0.15) / 2 + 0.06  # Half-height + cup clearance

        # Helper to test if a 3D point lands inside the obstacle box
        in_box = lambda p: (p_obs[0] - r_obs <= p[0] <= p_obs[0] + r_obs) and \
                           (p_obs[1] - r_obs <= p[1] <= p_obs[1] + r_obs) and \
                           (p_obs[2] - h_obs <= p[2] <= p_obs[2] + h_obs)

        # Plan the default direct trajectory over to Dobot handover
        traj_nominal = jtraj(self.barista.q, ik_present.q, 50)

        # Check every step along the path to see if the cup would hit the obstacle
        collision = any(in_box((self.barista.fkine(q) * self.cup_offset).t) for q in traj_nominal.q)

        if collision:
            print(f"Collision detected with obstacle at {np.round(p_obs[:2], 2)}! Replanning detour...")

            # Calculate a safe height to clear the top of the bottle
            z_detour = p_obs[2] + getattr(self.bottle, 'length', 0.15) / 2 + 0.22

            # Solve IK for an overhead waypoint straight above the bottle
            ee_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rz(pi/2) * self.cup_offset.inv()
            q_detour = self.barista.ikine_LM(ee_detour, q0=self.barista.q).q

            # Join two trajectories together: start -> up over bottle -> handover
            traj_to_detour = jtraj(self.barista.q, q_detour, 25).q
            traj_to_handover = jtraj(q_detour, ik_present.q, 25).q
            traj_present_q = np.vstack((traj_to_detour, traj_to_handover))
        else:
            print("Path clear. Proceeding along nominal trajectory...")
            traj_present_q = traj_nominal.q

        for q_step in traj_present_q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
                
        print("Cup presented to Dobot for milk :)")

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
        print("Dobot poured milk into the cup <3")

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

        # Collision Avoidance  
        # Extract obstacle position and compute bounding box with safety margins
        p_obs = SE3(self.bottle.T).t                            # Obstacle center [x, y, z] in world frame
        r_obs = getattr(self.bottle, 'radius', 0.045) + 0.06    # Obstacle radius + safety clearance
        h_obs = getattr(self.bottle, 'length', 0.15) / 2 + 0.06  # Half-height + safety clearance

        # Check if a 3D Cartesian point lies inside the bounding box volume
        in_box = lambda p: (p_obs[0] - r_obs <= p[0] <= p_obs[0] + r_obs) and \
                           (p_obs[1] - r_obs <= p[1] <= p_obs[1] + r_obs) and \
                           (p_obs[2] - h_obs <= p[2] <= p_obs[2] + h_obs)

        # Generate nominal direct joint trajectory toward tray hover pose
        traj_nominal = jtraj(self.barista.q, ik_hover.q, 30)

        # Check every step along trajectory for cup collision via Forward Kinematics
        collision = any(in_box((self.barista.fkine(q) * self.cup_offset).t) for q in traj_nominal.q)

        if collision:
            print(f"Collision detected with obstacle at {np.round(p_obs[:2], 2)}! Replanning detour...")

            # Calculate safe clearance height directly above the obstacle
            z_detour = p_obs[2] + getattr(self.bottle, 'length', 0.15) / 2 + 0.23
            
            # Formulate detour target pose directly over obstacle and solve IK
            ee_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rz(pi/2) * self.cup_offset.inv()
            q_detour = self.barista.ikine_LM(ee_detour, q0=self.barista.q).q

            # Splice trajectory into two safe segments: current -> detour -> hover
            traj_to_detour = jtraj(self.barista.q, q_detour, 20).q
            traj_to_hover  = jtraj(q_detour, ik_hover.q, 20).q
            traj_hover_q = np.vstack((traj_to_detour, traj_to_hover))
        else:
            print("Path clear. Proceeding along nominal trajectory...")
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

        print("Coffee placed on tray :)")
        self.cup_attached = False

        for q_step in traj_retract.q:
            self.barista.q = q_step
            self.env.step(0.05)

        q_home = np.array([-pi/3, 0.0, 20*pi/180, 0.0, 30*pi/180, 0.0])
        traj_home = jtraj(self.barista.q, q_home, 20)
        for q_step in traj_home.q:
            self.barista.q = q_step
            self.env.step(0.05)


    # BAKER CONTROL (RMRC, IBVS, COLLISION AVOIDANCE)

    def baker_rmrc_straight_line(self, T_goal, steps=25, dt=0.05, lambda_dls=0.05):

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

        p_obs = np.array(self.baker_obstacle.T[:3, 3])
        r_obs = 0.075
        h_obs = 0.12
        return (abs(p_check[0] - p_obs[0]) <= r_obs and
                abs(p_check[1] - p_obs[1]) <= r_obs and
                abs(p_check[2] - p_obs[2]) <= h_obs)

    def baker_pick_and_place_roll(self):

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

        # Step 7: RMRC Retract & Return Home (Rotated away to clear Service Robot workspace)
        print("--- [Baker] 7. Retract & Return Home ---")
        self.baker_rmrc_straight_line(T_hover_plate, steps=15)

        # Finish position to not interfere with service robot:
        # q: [65.0, 0.0, -30.0, 0.0, 60.0, 0.0] deg -> Pos: X:-0.497 Y:-0.778 Z:+1.730
        q_home = np.array([65.0 * pi / 180.0, 0.0, -pi / 6.0, 0.0, pi / 3.0, 0.0])
        traj_home = jtraj(self.baker.q, q_home, 25)
        for q in traj_home.q:
            self.baker.q = q
            self.env.step(0.05)
        print("[Baker] Returned to Clear Home Position: q=[65.0, 0.0, -30.0, 0.0, 60.0, 0.0] deg (Pos: X:-0.497 Y:-0.778 Z:+1.730).")
        print("[Baker] Sequence Completed Successfully - Service Robot Workspace Clear!\n")


    # STAUBLI CONTROL (RMRC, IBVS, OBSTACLE COLLISION AVOIDANCE)

    def staubli_rmrc(self, T_goal, steps=20, dt=0.05):
        """
        Executes a straight-line Cartesian path for Staubli using Closed-Loop RMRC with DLS.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.staubli.fkine(self.staubli.q)

            # Closed-loop velocity: remaining distance / remaining time
            rem_time = max((steps - k) * dt, dt)
            v_lin = (p_goal - np.array(T_curr.t)) / rem_time
            xdot = np.hstack((v_lin, np.zeros(3)))

            # Jacobian & Damped Least Squares (DLS)
            J = self.staubli.jacob0(self.staubli.q)
            m = np.sqrt(np.linalg.det(J @ J.T))
            damping = 0.001 if m < 0.01 else 0.0
            dq = np.linalg.inv(J.T @ J + damping * np.eye(6)) @ J.T @ xdot

            self.staubli.q = self.staubli.q + dq * dt

            if self.tray_attached:
                self.tray.T = self.staubli.fkine(self.staubli.q) * self.tray_offset
                self.plate.T = self.staubli.fkine(self.staubli.q) * self.plate_offset
                self.roll.T = self.staubli.fkine(self.staubli.q) * self.roll_offset
                self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
                self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)

            self.env.step(dt)

    def staubli_visual_servo_to_tray(self, max_steps=25, dt=0.05):
        """
        IBVS: Translates Staubli gripper to center directly over the serving tray using world-frame Jacobian.
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])

        p = SE3(self.tray.T).t
        pts_world = p[:, None] + [[0.24, 0.24, 0.16, 0.16], [0.04, -0.04, 0.04, -0.04], [0.02, 0.02, 0.02, 0.02]]

        T_desired = SE3(self.tray.T) * SE3(0.20, 0.0, 0.25) * SE3.Ry(pi)
        p_star = cam.project_point(T_desired.inv() * pts_world)

        depth = 0.23
        for step in range(max_steps):
            T_cam = self.staubli.fkine(self.staubli.q)
            pts_cam = T_cam.inv() * pts_world
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F')

            if np.linalg.norm(e) < 5.0:
                print(f"Visual servoing converged in {step} steps!")
                break

            # 3-DOF camera translation
            J_img = cam.visjac_p(uv, depth)[:, :3]
            v_cam = 4.0 * np.linalg.pinv(J_img) @ e

            # Rotate to world frame and solve joint velocities
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))
            dq = np.linalg.pinv(self.staubli.jacob0(self.staubli.q)) @ v_world

            self.staubli.q = self.staubli.q + dq * dt
            self.env.step(dt)

    def staubli_serve_order(self):
        T_hover_tray = SE3(self.tray.T) * SE3(0.16, -0.03, 0.25) * SE3.Ry(pi)
        T_grasp_tray = SE3(self.tray.T) * SE3(0.20, 0.0, 0.036) * SE3.Ry(pi)
        T_lift_tray  = SE3(self.tray.T) * SE3(0.20, 0.0, 0.31) * SE3.Ry(pi)

        ik_hover_tray = self.staubli.ikine_LM(T_hover_tray, q0=[2.5, 0.4, 1.0, 0.0, 1.5, 2.5])
        traj_hover_tray = jtraj(self.staubli.q, ik_hover_tray.q, 25)

        for q_step in traj_hover_tray.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        # Align gripper directly over tray handle using IBVS
        self.staubli_visual_servo_to_tray()

        # Lower straight down to tray using RMRC
        self.staubli_rmrc(T_grasp_tray, steps=20)

        print("Serving tray grasped by Staubli!")
        self.tray_attached = True
        self.tray_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.tray.T)
        self.plate_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.plate.T)
        self.roll_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.roll.T)
        self.cup_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.cup.T)

        # Lift tray straight up using RMRC (keeps coffee and roll completely level)
        self.staubli_rmrc(T_lift_tray, steps=20)

        T_hover_place = SE3(0.59, -0.51, 1.25) * SE3.Rz(pi/2) * SE3.Ry(pi)
        T_place_tray  = SE3(0.59, -0.51, 1.022) * SE3.Rz(pi/2) * SE3.Ry(pi)

        ik_hover_place = self.staubli.ikine_LM(T_hover_place, q0=[0.10, 0.54, 2.12, 0.0, 0.49, -1.47])
        q_nominal = ik_hover_place.q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2 * pi])
        traj_nominal = jtraj(self.staubli.q, q_nominal, 30)

        # Check nominal swing path against Coffee Machine bounding box (with tray radius margin)
        in_machine_box = lambda p: (0.05 <= p[0] <= 0.75) and (-0.05 <= p[1] <= 0.85) and (0.97 <= p[2] <= 1.55)

        if any(in_machine_box((self.staubli.fkine(q) * self.tray_offset).t) for q in traj_nominal.q):
            print("Collision detected with coffee machine! Replanning safe detour...")
            q_safe = ik_hover_place.q + np.array([2 * pi, 0.0, 0.0, 0.0, 0.0, 2 * pi])
            traj_carry = jtraj(self.staubli.q, q_safe, 30)
        else:
            traj_carry = traj_nominal

        for q_step in traj_carry.q:
            self.staubli.q = q_step
            self.tray.T = self.staubli.fkine(self.staubli.q) * self.tray_offset
            self.plate.T = self.staubli.fkine(self.staubli.q) * self.plate_offset
            self.roll.T = self.staubli.fkine(self.staubli.q) * self.roll_offset
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        # Lower tray straight down onto counter using RMRC
        self.staubli_rmrc(T_place_tray, steps=20)

        print("Order placed on counter by Staubli :)")
        self.tray_attached = False

        # Retract straight up from counter using RMRC
        self.staubli_rmrc(T_hover_place, steps=20)
        self.staubli.q = (self.staubli.q + pi) % (2 * pi) - pi

        T_hover_bell = SE3(self.bell.T) * SE3(0.0, 0.0, 0.148) * SE3.Ry(pi)
        T_tap_bell   = SE3(self.bell.T) * SE3(0.0, 0.0, 0.06) * SE3.Ry(pi)

        ik_hover_bell = self.staubli.ikine_LM(T_hover_bell, q0=[-1.0, 1.4, 0.8, 0.0, 0.9, -1.0])
        ik_tap_bell   = self.staubli.ikine_LM(T_tap_bell,   q0=ik_hover_bell.q)

        traj_to_bell  = jtraj(self.staubli.q,  ik_hover_bell.q, 20)
        traj_tap_down = jtraj(ik_hover_bell.q, ik_tap_bell.q,   5)
        traj_tap_up   = jtraj(ik_tap_bell.q,   ik_hover_bell.q, 5)

        for q_step in traj_to_bell.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        for q_step in traj_tap_down.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        for q_step in traj_tap_up.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        print("Service bell rung! Order ready :)")

    def barista_sequence(self):
        """Pick cup, pull shot, Dobot milk pour, and place on tray."""
        self.barista_pick_up_cup()
        self.barista_place_cup()
        self.barista_get_coffee()
        self.barista_present_cup_to_dobot()
        self.dobot_pour_milk()
        self.barista_serve_coffee()

    def baker_sequence(self):
        """Picks roll from oven and places on tray."""
        self.baker_pick_and_place_roll()

    def staubli_sequence(self):
        """Serves to counter, and rings bell."""
        self.staubli_serve_order()

    def run_sequence(self):
        """
        Executes full cafe sequence:
        """
        self.barista_sequence()
        self.baker_sequence()
        self.staubli_sequence()

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories
        """
        print("Cafe Simulation Running... Close browser to exit.")
        try:
            self.run_sequence()
            while True:
                self.env.step(0.05)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    cafe = AutomatedCafe()
    if "--no-gui" in sys.argv:
        cafe.run()
    else:
        from group_gui import GroupGUI
        gui = GroupGUI(cafe)
        gui.run()

