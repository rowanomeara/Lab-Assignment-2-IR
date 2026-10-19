import time
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
sys.modules['numpy.char'] = np.char 
from machinevisiontoolbox import CentralCamera

ROWAN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROWAN_DIR, ".."))
for path in (BASE_DIR, ROWAN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from roboticstoolbox.robot.Robot import Robot

if not hasattr(Robot, "fkine_geometry"):
    def _fkine_geometry(self, q, robot_alpha=1.0, collision_alpha=0.0):
        frames = self.fkine_all(q)
        return [frames[link.number] * SE3(g._T, check=False) for link in self.links for g in link.geometry]
    Robot.fkine_geometry = _fkine_geometry

# Robot Imports
from abb_irb120 import ABB_IRB120

try:
    from Jonas.Assignment2 import DoosanM0609
except Exception:
    DoosanM0609 = None

try:
    from Lachlan.LachlanMesh.staubli_tx60.staubliTX60 import StaubliTX60
except Exception:
    StaubliTX60 = None


class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment focused on Rowan's ABB IRB 120 Barista.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        self.estop_active = False
        self.sensor_breached = False
        self.is_running = False

        # Guard env.step globally against E-stop and safety sensor pauses
        orig_step = self.env.step
        def guarded_step(dt=0.05):
            self.update_safety_visuals()
            while self.estop_active or self.sensor_breached:
                self.update_safety_visuals()
                time.sleep(0.05)
            self.update_safety_visuals()
            orig_step(dt)
        self.env.step = guarded_step
        
        self.barista = ABB_IRB120()
        
        self.baker = DoosanM0609() if DoosanM0609 is not None else None
        self.staubli = StaubliTX60() if StaubliTX60 is not None else None
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add robots, tables, coffee equipment, props, and environment into Swift.
        All asset paths resolve relative to BASE_DIR.
        """
        base_path = BASE_DIR
        
        # Rowan's Barista Robot (ABB IRB 120)
        self.barista.base = SE3(-0.25, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.25, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)
        
        # Jonas's Baker Robot 
        if self.baker is not None:
            self.baker.base = SE3(-0.6, -1.0, 0.97)
            self.baker.base_link_mesh.T = SE3(-0.6, -1.0, 0.97)
            self.baker.q = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
            self.env.add(self.baker.base_link_mesh)
            self.env.add(self.baker)
        
        # Lachlan's Staubli Robot 
        if self.staubli is not None:
            self.staubli.base = SE3(0.30, -0.56, 0.97)
            self.staubli.base_link_mesh.T = SE3(0.30, -0.56, 0.97)
            self.env.add(self.staubli.base_link_mesh)
            self.env.add(self.staubli)
        
        # Dobot Magician 
        dobot_base = SE3(-0.9, 0.0, 0.97)
        self.dobot = DobotMagician(base=dobot_base)
        self.dobot.base = dobot_base
        
        # Level jug orientation for Dobot
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
        tabletop = Box(scale=[1.8, 2.2, 0.97], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.6, -0.2, 0.49)
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

        # Baked Rolls in Display Case
        roll_path = os.path.join(base_path, "Workcell meshes", "Baked Roll", "roll.stl")
        self.rolls = []
        for x in np.linspace(-0.15, 0.15, 4):
            for y in np.linspace(-1.270, -1.1, 5):
                roll = Mesh(roll_path, color=[0.82, 0.55, 0.28, 1.0])
                roll.T = SE3(x, y, 0.99) * SE3.Rx(-pi/2)
                self.env.add(roll)
                self.rolls.append(roll)
        
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

        # Base orientation for upright chair model: SE3.Ry(-pi/2) * SE3.Rz(-pi/2) faces -X
        chair_base_rot = SE3.Ry(-pi / 2) * SE3.Rz(-pi / 2)

        lounge_chair1 = part_mesh("chair")
        lounge_chair1.T = SE3(1.6, -2.1, 0.0) * SE3.Rz(-0.46 - pi) * chair_base_rot
        self.env.add(lounge_chair1)

        lounge_chair2 = part_mesh("chair")
        lounge_chair2.T = SE3(1.6, -2.9, 0.0) * SE3.Rz(0.46 - pi) * chair_base_rot
        self.env.add(lounge_chair2)

        # Potted Plants (Greenery Accents)
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
        # Barista Light Curtain (Coffee Machine front side)
        curtain_barista_p1 = part_mesh("SafetyLightCurtain")
        curtain_barista_p1.T = SE3(0.75, 0.05, 0.97)
        self.env.add(curtain_barista_p1)

        curtain_barista_p2 = part_mesh("SafetyLightCurtain")
        curtain_barista_p2.T = SE3(0.75, 0.80, 0.97)
        self.env.add(curtain_barista_p2)

        beam_barista = Box(scale=[0.01, 0.75, 0.35], color=[0.0, 0.8, 1.0, 0.25])
        beam_barista.T = SE3(0.75, 0.425, 1.15)
        self.env.add(beam_barista)

        # Barista Light Curtain (covering full outer flank side of counter and bench)
        curtain_flank_p1 = part_mesh("SafetyLightCurtain")
        curtain_flank_p1.T = SE3(0.74, 0.86, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_flank_p1)

        curtain_flank_p2 = part_mesh("SafetyLightCurtain")
        curtain_flank_p2.T = SE3(-1.45, 0.86, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_flank_p2)

        beam_flank = Box(scale=[2.19, 0.01, 0.35], color=[0.0, 0.8, 1.0, 0.25])
        beam_flank.T = SE3(-0.355, 0.86, 1.15)
        self.env.add(beam_flank)

        # Side Wing Light Curtain (covering full side counter wing)
        curtain_wing_p1 = part_mesh("SafetyLightCurtain")
        curtain_wing_p1.T = SE3(0.18, -1.75, 0.97) * SE3.Rz(pi/2)
        self.env.add(curtain_wing_p1)

        curtain_wing_p2 = part_mesh("SafetyLightCurtain")
        curtain_wing_p2.T = SE3(-0.80, -1.75, 0.97) * SE3.Rz(-pi/2)
        self.env.add(curtain_wing_p2)

        beam_wing = Box(scale=[0.98, 0.01, 0.35], color=[0.0, 0.8, 1.0, 0.25])
        beam_wing.T = SE3(-0.31, -1.75, 1.15)
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

        # Wall Shelf with Mounted Fire Extinguisher
        self.wall_shelf = Box(scale=[0.22, 0.30, 0.025], color=[0.35, 0.25, 0.18, 1.0])
        self.wall_shelf.T = SE3(-1.70, -1.0, 1.25)
        self.env.add(self.wall_shelf)

        fire_ext = part_mesh("FireExtinguisher")
        fire_ext.T = SE3(-1.68, -1.0, 1.27)
        self.env.add(fire_ext)

        #Caution Sign standing on floor
        warning_sign = part_mesh("WarningSign")
        warning_sign.T = SE3(-1.5, 0.7, 0.0)
        self.env.add(warning_sign)
        
        # Cleaning spray bottle (deliberate obstacle 
        self.bottle = Cylinder(radius=0.045, length=0.15, color=[0.9, 0.2, 0.2, 0.85])
        self.bottle.T = SE3(-0.50, -0.20, 0.97 + 0.075)
        self.env.add(self.bottle)

        # Visual bounding box
        r_box = getattr(self.bottle, 'radius', 0.045) + 0.02
        h_box = getattr(self.bottle, 'length', 0.15) + 0.04
        self.bottle_bbox = Box(scale=[2 * r_box, 2 * r_box, h_box], color=[1.0, 0.2, 0.2, 0.25])
        self.bottle_bbox.T = self.bottle.T
        self.env.add(self.bottle_bbox)

    @property
    def obstacle_box(self):
        """Dynamic axis-aligned bounding box [xmin, xmax, ymin, ymax, zmin, zmax] sticking to bottle."""
        p = SE3(self.bottle.T).t
        r = getattr(self.bottle, 'radius', 0.045) + 0.02
        h = getattr(self.bottle, 'length', 0.15) / 2 + 0.02
        return [p[0] - r, p[0] + r, p[1] - r, p[1] + r, p[2] - h, p[2] + h]

    def move_obstacle(self, x, y, z=None):
        """Move the obstacle bottle and automatically snap the visual bounding box to it."""
        z = (0.97 + 0.075) if z is None else z
        self.bottle.T = SE3(x, y, z)
        if hasattr(self, 'bottle_bbox'):
            self.bottle_bbox.T = self.bottle.T

    def update_safety_visuals(self):
        """Update 3D light curtain beams and status beacon colors based on safety state."""
        beam_color = [1.0, 0.0, 0.0, 0.55] if self.sensor_breached else [0.0, 0.8, 1.0, 0.25]
        for beam in getattr(self, 'curtain_beams', []):
            beam.color = beam_color
        if hasattr(self, 'beacon') and self.beacon is not None:
            if self.estop_active:
                self.beacon.color = [1.0, 0.0, 0.0, 0.95]
            elif self.sensor_breached:
                self.beacon.color = [1.0, 0.6, 0.0, 0.95]
            else:
                self.beacon.color = [0.0, 1.0, 0.0, 0.95]

    def press_physical_estop(self):
        """Simulates physical E-stop push button press."""
        self.estop_active = True
        self.update_safety_visuals()
        print("[SAFETY] Physical Emergency Stop pressed - robot halted.")

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
        #current_pose = self.barista.fkine(self.barista.q)
        target_pose = SE3(0.27, 0.22, 1.13) * SE3.Rx(-pi/2)
        
        self.rmrc_straight_line(target_pose, steps=25)
        
        print("Cup under coffee machine :)")
        self.cup_attached = False
        
        q_retract = self.barista.q + [0, -50 * pi / 180, 80 * pi / 180, 0, 0, 0]
        traj_retract = jtraj(self.barista.q, q_retract, 20)
            
        for q_step in traj_retract.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        
        coffee_color = [0.15, 0.08, 0.03, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        for q_step in jtraj(self.barista.q, ik_target.q, 25).q:
            self.barista.q = q_step
            self.env.step(0.05)
        self.cup_attached = True
        
        T_extract = SE3(-0.27, -0.20, 0.0) * self.barista.fkine(self.barista.q)
        self.rmrc_straight_line(T_extract, steps=25)
        
            
    def barista_present_cup_to_dobot(self):
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        ee_target = handover_cup_pose * self.cup_offset.inv()
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q)
        
        traj_present = jtraj(self.barista.q, ik_present.q, 50)
        for q_step in traj_present.q:
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

        self.milk = Cylinder(radius=0.058, length=0.015, color=[0.55, 0.36, 0.20, 1.0])
        self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
        self.env.add(self.milk)
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

        # Parametric collision avoidance: derive obstacle bounds and clearance dynamically from bottle
        if hasattr(self, 'bottle_bbox'):
            self.bottle_bbox.T = self.bottle.T
        p_obs = SE3(self.bottle.T).t
        r_obs = getattr(self.bottle, 'radius', 0.045) + 0.02
        h_obs = getattr(self.bottle, 'length', 0.15) / 2 + 0.02
        in_box = lambda p: (p_obs[0] - r_obs <= p[0] <= p_obs[0] + r_obs) and \
                           (p_obs[1] - r_obs <= p[1] <= p_obs[1] + r_obs) and \
                           (p_obs[2] - h_obs <= p[2] <= p_obs[2] + h_obs)

        traj_nominal = jtraj(self.barista.q, ik_hover.q, 30)

        if any(in_box((self.barista.fkine(q) * self.cup_offset).t) for q in traj_nominal.q):
            print(f"Collision detected with obstacle at {np.round(p_obs[:2], 2)}! Replanning detour...")
            z_detour = p_obs[2] + getattr(self.bottle, 'length', 0.15) / 2 + 0.23
            ee_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rz(pi/2) * self.cup_offset.inv()
            q_detour = self.barista.ikine_LM(ee_detour, q0=self.barista.q).q
            traj_hover_q = np.vstack((jtraj(self.barista.q, q_detour, 20).q, jtraj(q_detour, ik_hover.q, 20).q))
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
            
    def rmrc_straight_line(self, T_goal, steps=25, dt=0.05):
        """
        Executes a straight-line Cartesian path using Closed-Loop RMRC with DLS.
        """
        p_goal = np.array(T_goal.t)

        for k in range(steps):
            T_curr = self.barista.fkine(self.barista.q)
            
            # Closed-loop velocity: remaining distance / remaining time (eliminates undershoot)
            rem_time = max((steps - k) * dt, dt) # Remaining time to goal
            v_lin = (p_goal - np.array(T_curr.t)) / rem_time # Linear velocity vector: straight line from current position to goal
            xdot = np.hstack((v_lin, np.zeros(3))) # 6D Cartesian velocity: move linearly, but ZERO angular rotation (keep cup level)

            # Jacobian & DLS
            J = self.barista.jacob0(self.barista.q) # 1. Get the 6x6 Jacobian at current arm position
            m = np.sqrt(np.linalg.det(J @ J.T)) # 2. Check manipulability (distance to singularity)
            damping = 0.001 if m < 0.01 else 0.0 # 3. Apply Damped Least Squares if manipulability drops below threshold
            dq = np.linalg.inv(J.T @ J + damping * np.eye(6)) @ J.T @ xdot # 4. Solve for joint speeds

            self.barista.q += dq * dt # Numerical integration (Euler step): update joint angles

            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                if hasattr(self, 'coffee'):
                    self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)

            self.env.step(dt)
    
    
    def visual_servo_to_cup(self, max_steps=25, dt=0.05):
        """
        IBVS: Translates gripper to center directly over the cup using world-frame Jacobian.
        """
        cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512]) #Instantiate virtual camera (Lab 8 parameters)

        p = SE3(self.cup.T).t
        pts_world = p[:, None] + [[0.04, -0.04, 0.04, -0.04], [0, 0, 0, 0], [0.08, 0.08, 0.01, 0.01]] #Define 4 physical feature points on the cup in world space 
        #(front face: top-left, top-right, bottom-left, bottom-right)

        T_desired = SE3(self.cup.T) * SE3(0.0, -0.15, 0.05) * SE3.Rx(-pi/2) #Create the reference image (what the camera sees when perfectly aligned)
        p_star = cam.project_point(T_desired.inv() * pts_world) #This gives p_star the target pixel coordinates

        depth = 0.15
        for step in range(max_steps):
            T_cam = self.barista.fkine(self.barista.q) # Get current camera pose in world
            pts_cam = T_cam.inv() * pts_world # Transform 3D points from world to current camera frame, then project to pixels
            uv = cam.project_point(pts_cam)
            e = (p_star - uv).flatten('F') # Calculate pixel error vector

            # Reads current camera pose T_cam
            # Converts world points into the camera's local coordinate frame
            # Projects them through the lens into pixel coordinates
            # Subtracts current pixels from target pixels (e = p_star - uv)
            if np.linalg.norm(e) < 5.0:
                print(f"Visual servoing converged in {step} steps!")
                break
            #If the total error is under 5 pixels (equivalent to <0.5 mm physical error), the arm is centered. We stop servoing and move to the grasp.

            # 3-DOF camera translation
            J_img = cam.visjac_p(uv, depth)[:, :3] #Image Jacobian converts pixel error into camera velocity
            v_cam = 0.8 * np.linalg.pinv(J_img) @ e

            # Rotate to world frame so up is always up (+Z), right is always right (+X)
            v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))
            dq = np.linalg.pinv(self.barista.jacob0(self.barista.q)) @ v_world
            
            #Apply joint step to robot
            self.barista.q += dq * dt
            self.env.step(dt)

    def run_sequence(self):
        """
        Executes the full automated barista sequence.
        """
        self.barista_pick_up_cup()
        self.barista_place_cup()
        self.barista_get_coffee()
        self.barista_present_cup_to_dobot()
        self.dobot_pour_milk()
        self.barista_serve_coffee()

    def run(self):
        """
        Main execution loop for Rowan's Barista robot and Dobot milk pouring interaction.
        """
        print("Rowan's Barista Simulation Running... Close browser to exit.")
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
        from barista_gui import BaristaGUI
        gui = BaristaGUI(cafe)
        gui.run()
