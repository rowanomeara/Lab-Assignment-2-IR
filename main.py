import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh
from spatialmath import SE3
from ir_support.robots import DobotMagician
from ir_support_extra_parts.parts import part_mesh 
import os

# Import
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

        #self.dispatcher = Laclan's tx60
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

        #Baker Robot
        self.baker.base = SE3(-0.25, 0.4, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.25, 0.4, 0.97)
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)

        
        #Dobot 
        self.dobot = DobotMagician()
        self.dobot.base = SE3(0.0, -0.5, 0.97) #* SE3.Rz(-pi/4)
        self.dobot.add_to_env(self.env)
        
        import os
        from spatialgeometry import Box
        base_path = os.path.dirname(os.path.abspath(__file__))

        #Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)
        
        # Plank
        tabletop = Box(scale=[1.2, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.3, -0.2, 0.95)
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
        
        #Tray
        self.tray = part_mesh("Tray")
        self.tray.T = SE3(-0.55, -0.5, 0.97)
        self.env.add(self.tray)
        
    def barista_pick_up_cup(self):
        from roboticstoolbox import jtraj
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.10, 0.23) * SE3.Rx(-pi/2)
        
        ik_hover  = self.barista.ikine_LM(hover_pose,  q0=self.barista.q)
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise  = self.barista.ikine_LM(hover_pose,  q0=ik_target.q)   
        
        traj_hover  = jtraj(self.barista.q, ik_hover.q, 10)
        traj_target = jtraj(ik_hover.q,     ik_target.q, 5)
        traj_raise  = jtraj(ik_target.q,    ik_raise.q, 5)
        
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
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 10)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, 0], 10)
            
        print("Cup under coffee machine :)")
        self.cup_attached = False
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        from roboticstoolbox import jtraj
        from spatialgeometry import Cylinder
        
        #Coffee liquid
        coffee_color = [0.35, 0.18, 0.05, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 20)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, -25*pi/180], 20)
            
        print("Cup under coffee machine :)")
        self.cup_attached = True
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
        traj_rot  = jtraj(self.barista.q,  self.barista.q + [-90*pi/180, 0, 0, 0, 0, 0], 30)
        
        for q_step in traj_rot.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            

 

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        
        self.barista_pick_up_cup()
        
        self.barista_place_cup()
        
        self.barista_get_coffee()
        
        while True:
            self.env.step(0.05)

if __name__ == "__main__":

    cafe = AutomatedCafe()
    cafe.run()
