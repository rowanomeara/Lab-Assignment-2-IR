import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh, Box, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj
from ir_support.robots import DobotMagician
from ir_support_extra_parts.parts import part_mesh 
import os
import sys
import time
sys.modules['numpy.char'] = np.char
from machinevisiontoolbox import CentralCamera

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

        self.estop_active = False
        self.sensor_breached = False
        self.is_running = False
        self.tray_attached = False

        # Guard env.step globally against E-stop and safety sensor pauses
        orig_step = self.env.step
        def guarded_step(dt=0.05):
            while self.estop_active or self.sensor_breached:
                time.sleep(0.05)
            orig_step(dt)
        self.env.step = guarded_step
        
        self.barista = ABB_IRB120()
        
        self.baker = DoosanM0609()
        
        self.staubli = StaubliTX60()
        
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

        # Cafe Lounge Seating: Tufted Leather Sofa & Round Marble Coffee Table
        sofa_path = os.path.join(base_path, "Workcell meshes", "Sofa", "sofa.stl")
        sofa = Mesh(sofa_path, color=[0.36, 0.22, 0.14, 1.0])
        sofa.T = SE3(3.4, -2.5, 0.0) * SE3.Rz(-pi / 2) * SE3.Rx(pi / 2)
        self.env.add(sofa)

        coffee_table_path = os.path.join(base_path, "Workcell meshes", "CoffeeTable", "coffee_table.stl")
        coffee_table = Mesh(coffee_table_path, color=[0.94, 0.93, 0.90, 1.0], scale=[0.85, 0.85, 0.85])
        coffee_table.T = SE3(2.4, -2.5, 0.0) * SE3.Rx(pi / 2)
        self.env.add(coffee_table)

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

        # High-Top Bistro & Dining Area
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

        # Customers / Patrons
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
        curtain_barista_p1 = part_mesh("SafetyLightCurtain")
        curtain_barista_p1.T = SE3(0.75, 0.05, 0.97)
        self.env.add(curtain_barista_p1)

        curtain_barista_p2 = part_mesh("SafetyLightCurtain")
        curtain_barista_p2.T = SE3(0.75, 0.80, 0.97)
        self.env.add(curtain_barista_p2)

        beam_barista = Box(scale=[0.01, 0.75, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_barista.T = SE3(0.75, 0.425, 1.36)
        self.env.add(beam_barista)

        curtain_flank_p1 = part_mesh("SafetyLightCurtain")
        curtain_flank_p1.T = SE3(0.74, 0.86, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_flank_p1)

        curtain_flank_p2 = part_mesh("SafetyLightCurtain")
        curtain_flank_p2.T = SE3(0.15, 0.86, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_flank_p2)

        beam_flank = Box(scale=[0.59, 0.01, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_flank.T = SE3(0.445, 0.86, 1.36)
        self.env.add(beam_flank)

        curtain_wing_p1 = part_mesh("SafetyLightCurtain")
        curtain_wing_p1.T = SE3(0.18, -1.75, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_wing_p1)

        curtain_wing_p2 = part_mesh("SafetyLightCurtain")
        curtain_wing_p2.T = SE3(-0.80, -1.75, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_wing_p2)

        beam_wing = Box(scale=[0.98, 0.01, 0.76], color=[0.0, 0.8, 1.0, 0.25])
        beam_wing.T = SE3(-0.31, -1.75, 1.36)
        self.env.add(beam_wing)

        self.curtain_beams = [beam_barista, beam_flank, beam_wing]

        light_tower = part_mesh("LightTower")
        light_tower.T = SE3(-0.05, -1.65, 0.97)
        self.env.add(light_tower)

        self.beacon = Cylinder(radius=0.03, length=0.07, color=[0.0, 1.0, 0.0, 0.95])
        self.beacon.T = SE3(-0.05, -1.65, 1.45)
        self.env.add(self.beacon)

        barrier = part_mesh("barrier1.5x0.2x1m")
        barrier.T = SE3(-1.5, 0.5, 0.0) * SE3.Rz(pi/2)
        self.env.add(barrier)

        self.wall_shelf = Box(scale=[0.22, 0.30, 0.025], color=[0.35, 0.25, 0.18, 1.0])
        self.wall_shelf.T = SE3(-1.70, -1.0, 1.25)
        self.env.add(self.wall_shelf)

        fire_ext = part_mesh("FireExtinguisher")
        fire_ext.T = SE3(-1.68, -1.0, 1.27)
        self.env.add(fire_ext)

        warning_sign = part_mesh("WarningSign")
        warning_sign.T = SE3(-1.5, 0.7, 0.0)
        self.env.add(warning_sign)

    def press_physical_estop(self):
        """Simulates physical E-stop push button press."""
        self.estop_active = True
        print("[SAFETY] Physical Emergency Stop pressed - robot halted.")
        
        
        
    def barista_pick_up_cup(self):
        from roboticstoolbox import jtraj
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.10, 0.23) * SE3.Rx(-pi/2)
        
        ik_hover  = self.barista.ikine_LM(hover_pose,  q0=self.barista.q)
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise  = self.barista.ikine_LM(hover_pose,  q0=ik_target.q)   
        
        traj_hover  = jtraj(self.barista.q, ik_hover.q, 20)
        traj_target = jtraj(ik_hover.q,     ik_target.q, 20)
        traj_raise  = jtraj(ik_target.q,    ik_raise.q, 20)
        
        #Move to Hover
        for q_step in traj_hover.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
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
        from roboticstoolbox import jtraj
        
        current_pose = self.barista.fkine(self.barista.q)
        target_pose = SE3(0.27, -0.28, -0.07) * current_pose
        
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 20)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, 0], 20)
            
        print("Cup under coffee machine :)")
        self.cup_attached = False
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        from roboticstoolbox import jtraj
        from spatialgeometry import Cylinder
        
        coffee_color = [0.15, 0.08, 0.03, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 30)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, -25*pi/180], 30)
            
        print("Cup under coffee machine :)")
        self.cup_attached = True
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
        traj_rot  = jtraj(self.barista.q,  self.barista.q + [-100*pi/180, 0, 0, 0, 0, 0], 30)
        
        for q_step in traj_rot.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
    def barista_present_cup_to_dobot(self):
        from roboticstoolbox import jtraj
        
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        
        ee_target = handover_cup_pose * self.cup_offset.inv()
        
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q, slimit=1)
        q_present = (ik_present.q + pi) % (2 * pi) - pi
        
        traj_present = jtraj(self.barista.q, q_present, 30)
        for q_step in traj_present.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
                
        print("Cup presented to Dobot for milk :)")

    def dobot_pour_milk(self):
        from roboticstoolbox import jtraj
        from spatialgeometry import Cylinder

        def move(q_target, steps):
            for q in jtraj(self.dobot.q, q_target, steps).q:
                self.dobot.q = q
                self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
                self.env.step(0.05)

        T_cup_rim = (SE3(self.cup.T) * SE3(-0.05, 0.0, 0.125)).t
        T_ee_target = SE3(T_cup_rim - [0.156, 0.0, 0.080]) * SE3.Rx(pi)

        sol_reach = self.dobot.ikine_LM(T_ee_target, q0=self.dobot.q, mask=[1, 1, 1, 0, 0, 0], slimit=1)
        q_reach = (sol_reach.q + pi) % (2 * pi) - pi
        q_reach[3] = -(q_reach[1] + q_reach[2])

        move(q_reach, 20)

        q_pour = q_reach.copy()
        q_pour[3] += 25 * pi / 180
        move(q_pour, 15)

        self.milk = Cylinder(radius=0.058, length=0.015, color=[0.55, 0.36, 0.20, 1.0])
        self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
        self.env.add(self.milk)
        print("Dobot poured milk into the cup <3")

        move(q_reach, 15)
        move(self.dobot_q_ready, 20)
        
    def barista_serve_coffee(self):
        from roboticstoolbox import jtraj

        target_cup = SE3(self.tray.T) * SE3(0.08, 0.0, 0.03)
        hover_cup = SE3(0, 0, 0.12) * target_cup

        ee_hover = hover_cup * SE3.Rz(pi) * self.cup_offset.inv()
        ee_target = target_cup * SE3.Rz(pi) * self.cup_offset.inv()

        ik_hover = self.barista.ikine_LM(ee_hover, q0=[-1.48, 0.74, 0.60, 0.10, -1.34, 1.55])
        ik_serve = self.barista.ikine_LM(ee_target, q0=ik_hover.q)
        ik_retract = self.barista.ikine_LM(ee_hover, q0=ik_serve.q)

        traj_hover = jtraj(self.barista.q, ik_hover.q, 30)
        traj_serve = jtraj(ik_hover.q, ik_serve.q, 15)
        traj_retract = jtraj(ik_serve.q, ik_retract.q, 15)

        for q_step in traj_hover.q:
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

    def baker_pick_and_place_roll(self):
        """
        Baker robot picks up the baked roll and places it onto the plate on the serving tray.
        """
        p_roll = np.array(self.roll.T[:3, 3])
        p_plate = np.array(self.plate.T[:3, 3])

        T_hover_roll = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.15) * SE3.Rx(-pi)
        T_grasp_roll = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.04) * SE3.Rx(-pi)
        T_hover_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.15) * SE3.Rx(-pi)
        T_drop_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.04) * SE3.Rx(-pi)

        q_seed = np.array([pi/4, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        ik_hover_roll = self.baker.ikine_LM(T_hover_roll, q0=q_seed)
        ik_grasp_roll = self.baker.ikine_LM(T_grasp_roll, q0=ik_hover_roll.q)
        ik_lift_roll = self.baker.ikine_LM(T_hover_roll, q0=ik_grasp_roll.q)
        ik_hover_plate = self.baker.ikine_LM(T_hover_plate, q0=ik_lift_roll.q)
        ik_drop_plate = self.baker.ikine_LM(T_drop_plate, q0=ik_hover_plate.q)
        ik_retract = self.baker.ikine_LM(T_hover_plate, q0=ik_drop_plate.q)

        # 1. Approach above roll
        traj1 = jtraj(self.baker.q, ik_hover_roll.q, 25)
        for q in traj1.q:
            self.baker.q = q
            self.env.step(0.05)

        # 2. Lower to roll
        traj2 = jtraj(ik_hover_roll.q, ik_grasp_roll.q, 15)
        for q in traj2.q:
            self.baker.q = q
            self.env.step(0.05)

        print("Baked roll grasped by Baker!")
        self.roll_attached = True
        self.roll_offset = self.baker.fkine(self.baker.q).inv() * SE3(self.roll.T)

        # 3. Lift roll
        traj3 = jtraj(ik_grasp_roll.q, ik_lift_roll.q, 15)
        for q in traj3.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        # 4. Transit over plate on serving tray
        traj4 = jtraj(ik_lift_roll.q, ik_hover_plate.q, 25)
        for q in traj4.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        # 5. Lower roll onto plate
        traj5 = jtraj(ik_hover_plate.q, ik_drop_plate.q, 15)
        for q in traj5.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        print("Baked roll placed on plate atop serving tray :)")
        self.roll_attached = False

        # 6. Retract from plate
        traj6 = jtraj(ik_drop_plate.q, ik_retract.q, 15)
        for q in traj6.q:
            self.baker.q = q
            self.env.step(0.05)

        # 7. Return to home configuration
        q_home = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
        traj_home = jtraj(self.baker.q, q_home, 25)
        for q in traj_home.q:
            self.baker.q = q
            self.env.step(0.05)

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

        for q_step in traj_tap_down.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        for q_step in traj_tap_up.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        print("Service bell rung! Order ready :)")

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        
        self.barista_pick_up_cup()
        
        self.barista_place_cup()
        
        self.barista_get_coffee()

        self.barista_present_cup_to_dobot()

        self.dobot_pour_milk()
        
        self.barista_serve_coffee()

        self.baker_pick_and_place_roll()

        self.staubli_serve_order()
        
        while True:
            self.env.step(0.05)

if __name__ == "__main__":

    cafe = AutomatedCafe()
    cafe.run()
