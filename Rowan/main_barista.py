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
        
        # Primary Robot: Rowan's ABB IRB 120
        self.barista = ABB_IRB120()
        
        # Static teammate robots (visual background in shared workcell)
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
        
        # Environment Props
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
        chair.T = SE3(2.5, 2.5, 0.0) * SE3.Ry(-pi / 2) * SE3.Rz(-pi / 2)
        self.env.add(chair)
        
        customer = part_mesh("personMaleCasual") 
        customer.T = SE3(1.5, 0.5, 0.0) * SE3.Rz(pi)
        self.env.add(customer)
        
        takeaway_cup = part_mesh("TakeawayCup")
        takeaway_cup.T = SE3(0.5, -0.1, 0.97)
        self.env.add(takeaway_cup)
        
        self.estop = part_mesh("emergencyStopWallMounted")
        self.estop.T = SE3(-1.77, -0.6, 1.4) * SE3.Rz(-pi/2)
        self.env.add(self.estop)

    def barista_pick_up_cup(self):
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.25, 0.16) * SE3.Rx(-pi/2)
        
        ik_hover  = self.barista.ikine_LM(hover_pose,  q0=self.barista.q)
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise  = self.barista.ikine_LM(hover_pose,  q0=ik_target.q)   
        
        traj_hover  = jtraj(self.barista.q, ik_hover.q, 20)
        traj_target = jtraj(ik_hover.q,     ik_target.q, 20)
        traj_raise  = jtraj(ik_target.q,    ik_raise.q, 20)
        
        # Move to Hover
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
        # 1. Target pose inside the machine bay relative to current pose
        current_pose = self.barista.fkine(self.barista.q)
        target_pose = SE3(0.27, -0.12, 0.0) * current_pose
        
        # 2. Pure straight-line insertion using RMRC (replaces ikine_LM + jtraj)
        self.rmrc_straight_line(target_pose, steps=25)
        
        print("Cup under coffee machine :)")
        self.cup_attached = False
        
        # 3. Retract the empty gripper up and away from the cup
        q_retract = self.barista.q + [0, -50 * pi / 180, 80 * pi / 180, 0, 0, 0]
        traj_retract = jtraj(self.barista.q, q_retract, 20)
            
        for q_step in traj_retract.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        
        # 1. Fill cup with coffee
        coffee_color = [0.15, 0.08, 0.03, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)
        
        # 2. Regrasp the cup under the tap
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        for q_step in jtraj(self.barista.q, ik_target.q, 25).q:
            self.barista.q = q_step
            self.env.step(0.05)
        self.cup_attached = True
        
        # 3. Step 1: Straight horizontal RMRC extraction out of machine bay
        T_extract = SE3(-0.27, -0.20, 0.0) * self.barista.fkine(self.barista.q)
        self.rmrc_straight_line(T_extract, steps=25)
        
        
            
    def barista_present_cup_to_dobot(self):
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        ee_target = handover_cup_pose * self.cup_offset.inv()
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q)
        
        traj_present = jtraj(self.barista.q, ik_present.q, 30)
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

            # Jacobian & DLS (tuned for IRB 120 manipulability scale)
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

    def run(self):
        """
        Main execution loop for Rowan's Barista robot and Dobot milk pouring interaction.
        """
        print("Rowan's Barista Simulation Running... Close browser to exit.")
        try:
            self.barista_pick_up_cup()
            self.barista_place_cup()
            self.barista_get_coffee()
            self.barista_present_cup_to_dobot()
            self.dobot_pour_milk()
            self.barista_serve_coffee()
            
            while True:
                self.env.step(0.05)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    cafe = AutomatedCafe()
    cafe.run()
